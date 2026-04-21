import os
import sys
import signal

PID_FILE = "/tmp/groq_audio_daemon.pid"

FIX_FLAG_FILE = "/tmp/groq_audio_fixon.flag"

# Quick toggle path that doesn't require any third-party libraries
if len(sys.argv) > 1 and sys.argv[1] in ("toggle", "toggle_sys"):
    if "fixon" in sys.argv:
        open(FIX_FLAG_FILE, "w").close()

    try:
        with open(PID_FILE, "r") as f:
            pid = int(f.read().strip())

        if sys.argv[1] == "toggle":
            os.kill(pid, signal.SIGUSR1)
            print(f"Sent toggle (MIC) signal to daemon (PID: {pid})")
        elif sys.argv[1] == "toggle_sys":
            os.kill(pid, signal.SIGUSR2)
            print(f"Sent toggle (SYS) signal to daemon (PID: {pid})")

    except FileNotFoundError:
        print("Daemon is not running. Start it first without arguments.")
        sys.exit(1)
    except ProcessLookupError:
        print("Daemon process not found. It might have crashed. Restart it.")
        if os.path.exists(PID_FILE):
            os.remove(PID_FILE)
        sys.exit(1)
    except Exception as e:
        print(f"Error signaling daemon: {e}")
        sys.exit(1)
    sys.exit(0)

import asyncio
import io
import struct
import logging
import subprocess
from collections import deque

import aiohttp
import numpy as np
import av

from dotenv import load_dotenv

# Load environment variables from .env file relative to this script
script_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(script_dir, ".env")
load_dotenv(env_path)

UI_LANGUAGE = os.environ.get("UI_LANGUAGE", "en").lower()

STRINGS = {
    "en": {
        "rec_mic_title": "🎙️ Recording (Microphone)",
        "rec_mic_msg": "Speak now...",
        "rec_sys_title": "🔊 Recording (System)",
        "rec_sys_msg": "Capturing system audio...",
        "processing_title": "🛑 Processing...",
        "processing_msg": "Waiting for API response",
        "silence_title": "🔇 Silence",
        "silence_msg": "No voice detected",
        "success_title": "✅ Processed",
        "error_title": "❌ Error",
        "error_msg": "Failed to transcribe text",
        "sys_error_title": "❌ System Error"
    },
    "ru": {
        "rec_mic_title": "🎙️ Запись (Микрофон)",
        "rec_mic_msg": "Говорите...",
        "rec_sys_title": "🔊 Запись (Система)",
        "rec_sys_msg": "Захват системного звука...",
        "processing_title": "🛑 Обработка...",
        "processing_msg": "Ждем ответ от API",
        "silence_title": "🔇 Тишина",
        "silence_msg": "Голос не обнаружен",
        "success_title": "✅ Запись обработана",
        "error_title": "❌ Ошибка",
        "error_msg": "Не удалось распознать текст",
        "sys_error_title": "❌ Системная Ошибка"
    }
}

def get_string(key):
    lang = UI_LANGUAGE if UI_LANGUAGE in STRINGS else "en"
    return STRINGS[lang].get(key, STRINGS["en"].get(key, key))

TARGET_RMS = 0.05
MAX_GAIN = 1.5
RATE_WHISPER = 16000
CHUNK_VAD_16K = 512

_vad_session = None

def get_vad_session():
    """Initializes the Silero VAD session."""
    global _vad_session
    if _vad_session is None:
        try:
            import onnxruntime

            vad_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "silero_vad.onnx"
            )
            _vad_session = onnxruntime.InferenceSession(vad_path)
        except Exception as e:
            logging.warning(f"silero_vad.onnx not loaded. Error: {e}")
            _vad_session = None
    return _vad_session


class TrueRMSLimiter:
    """Limits audio level using True RMS and smoothing."""

    def __init__(self, target_rms, max_gain, smoothing=0.85):
        self.target_rms = target_rms
        self.max_gain = max_gain
        self.smoothing = smoothing
        self.smooth_gain = 1.0

    def process(self, chunk):
        """Processes an audio chunk to normalize gain."""
        rms = np.sqrt(np.mean(chunk**2) + 1e-8)
        target_g = self.target_rms / rms
        target_g = np.clip(target_g, 1.0, self.max_gain)
        self.smooth_gain = (
            self.smoothing * self.smooth_gain
            + (1.0 - self.smoothing) * target_g
        )
        return np.tanh(chunk * self.smooth_gain)


class ONNXVAD:
    """Voice Activity Detection using Silero ONNX model."""

    def __init__(
        self, session, threshold=0.7, min_silence_ms=200, chunk_ms=32
    ):
        self.session = session
        self.threshold = threshold
        self.min_silence_chunks = int(min_silence_ms / chunk_ms)
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        self.context = np.zeros((1, 64), dtype=np.float32)
        self.sr = np.array([16000], dtype=np.int64)
        self.triggered = False
        self.silence_counter = 0

    def __call__(self, chunk_16k):
        """Processes a 16kHz chunk and returns 'start', 'end', or None."""
        x = np.concatenate((self.context, chunk_16k.reshape(1, -1)), axis=1)
        ort_outs = self.session.run(
            None, {"input": x, "state": self.state, "sr": self.sr}
        )
        prob = ort_outs[0][0][0]
        self.state = ort_outs[1]
        self.context = chunk_16k[-64:].reshape(1, 64)

        if prob >= self.threshold:
            self.silence_counter = 0
            if not self.triggered:
                self.triggered = True
                return "start"
        else:
            if self.triggered:
                self.silence_counter += 1
                if self.silence_counter >= self.min_silence_chunks:
                    self.triggered = False
                    self.silence_counter = 0
                    return "end"
        return None


def _make_wav(audio_float32: np.ndarray, rate: int) -> bytes:
    """Wraps raw float32 audio into a WAV container (int16)."""
    audio_int16 = (
        (audio_float32 * 32767.0).clip(-32768, 32767).astype(np.int16)
    )
    data_bytes = audio_int16.tobytes()
    return (
        struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF",
            36 + len(data_bytes),
            b"WAVE",
            b"fmt ",
            16,
            1,
            1,
            rate,
            rate * 2,
            2,
            16,
            b"data",
            len(data_bytes),
        )
        + data_bytes
    )


def wav_to_ogg(wav_bytes: bytes) -> bytes:
    """Converts a WAV byte string to an OGG OPUS byte string for Telegram Voice Notes."""
    in_io = io.BytesIO(wav_bytes)
    out_io = io.BytesIO()
    try:
        with av.open(in_io) as in_container:
            in_stream = in_container.streams.audio[0]
            with av.open(out_io, mode="w", format="ogg") as out_container:
                out_stream = out_container.add_stream("libopus", rate=48000)
                for frame in in_container.decode(in_stream):
                    for packet in out_stream.encode(frame):
                        out_container.mux(packet)
                for packet in out_stream.encode(None):
                    out_container.mux(packet)
        return out_io.getvalue()
    except Exception as e:
        logging.error(f"Error converting wav to ogg: {e}")
        return wav_bytes




async def fix_text_with_llm(text: str) -> str:
    """Uses Groq's LLM to slightly fix grammar and add paragraphs to the transcribed text."""
    groq_api_key = os.environ.get("GROQ_API_KEY", "")
    if not groq_api_key:
        return text

    headers = {
        "Authorization": f"Bearer {groq_api_key}",
        "Content-Type": "application/json"
    }

    # Use the model the user requested explicitly.
    model = os.environ.get("LLM_MODEL", "openai/gpt-oss-120b")
    # Allow overriding the base URL in case they use OpenRouter or another provider for the LLM fix
    llm_base_url = os.environ.get("LLM_BASE_URL", "https://api.groq.com/openai/v1/chat/completions")

    system_prompt = (
        "Your task is to slightly fix grammar and divide the text into paragraphs. "
        "Please note that the text is a transcribed voice memo, so it may contain "
        "miswordings or phonetic errors. Account for this in your corrections. "
        "Make minimal changes. Do not remake the text fully. Respond ONLY with the fixed text."
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ],
        "temperature": 0.2
    }

    try:
        async with aiohttp.ClientSession() as session_http:
            async with session_http.post(
                llm_base_url,
                headers=headers,
                json=payload,
            ) as resp:
                if resp.status != 200:
                    err = await resp.text()
                    logging.error(f"LLM API Error {resp.status}: {err}")
                    return text
                json_resp = await resp.json()
                fixed_text = json_resp.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                return fixed_text if fixed_text else text
    except Exception as e:
        logging.error(f"Error calling Groq LLM API: {e}")
        return text


async def process_ready_audio(file_data: bytes) -> tuple[str, bytes]:
    """
    Sends the pre-processed audio bytes to Groq Whisper.
    """
    if not file_data:
        return "", b""

    # Replaced config with env vars as requested.
    groq_api_key = os.environ.get("GROQ_API_KEY", "")
    whisper_model = os.environ.get("WHISPER_MODEL", "whisper-large-v3")
    bot_language = os.environ.get("BOT_LANGUAGE", "ru")

    headers = {"Authorization": f"Bearer {groq_api_key}"}
    data = aiohttp.FormData()
    data.add_field("model", whisper_model)
    data.add_field("response_format", "json")
    data.add_field("temperature", "0")
    if bot_language:
        # Pass language hint to Whisper (ISO-639-1)
        data.add_field("language", bot_language.lower()[:2])

    data.add_field(
        "file",
        file_data,
        filename="processed_audio.ogg",
        content_type="audio/ogg",
    )

    try:
        async with aiohttp.ClientSession() as session_http:
            api_url = os.environ.get("GROQ_API_URL", "https://api.groq.com/openai/v1/audio/transcriptions")
            async with session_http.post(
                api_url,
                headers=headers,
                data=data,
            ) as resp:
                if resp.status != 200:
                    err = await resp.text()
                    logging.error(
                        f"Groq Whisper API Error {resp.status}: {err}"
                    )
                    return "", b""
                json_resp = await resp.json()
                text = json_resp.get("text", "").strip()
                return text, file_data
    except Exception as e:
        logging.error(f"Error calling Groq API: {e}")
        return "", b""

ID_FILE = "/tmp/groq_notif.id"

def play_sound(sound_path):
    """Plays a sound using paplay, killing any existing paplay instances to prevent overlap issues."""
    subprocess.run(["pkill", "-x", "paplay"], stderr=subprocess.DEVNULL)
    subprocess.Popen(
        ["paplay", sound_path],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

def send_notification(title, message, replaces_id=None):
    cmd = ["notify-send", "-p"]
    if replaces_id:
        cmd.extend(["-r", str(replaces_id)])
    cmd.extend([title, message])
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return None

def close_notification(notif_id):
    if not notif_id:
        return
    cmd = [
        "busctl", "--user", "call", "org.freedesktop.Notifications",
        "/org/freedesktop/Notifications", "org.freedesktop.Notifications",
        "CloseNotification", "u", str(notif_id)
    ]
    subprocess.run(cmd, stderr=subprocess.DEVNULL)

class AudioDaemon:
    def __init__(self):
        self.is_recording = False
        self.process = None
        self.audio_chunks = []
        self.read_task = None
        self.processing_task = None
        self.timeout_task = None

        # Real-time processing state
        self.vad = None
        self.limiter = None
        self.speech_buffer = []
        self.history_q = None
        self.state_speech = False
        self.chunk_remainder = b""

    async def _recording_timeout(self):
        # Wait for 10 minutes (600 seconds)
        await asyncio.sleep(600)
        if self.is_recording:
            logging.info("10-minute recording limit reached. Automatically stopping.")
            self.stop_recording()

    async def _read_stdout(self):
        chunk_size_bytes = CHUNK_VAD_16K * 2

        while self.process and not self.process.stdout.at_eof():
            chunk = await self.process.stdout.read(4096)
            if not chunk:
                continue

            self.chunk_remainder += chunk

            while len(self.chunk_remainder) >= chunk_size_bytes:
                chunk_bytes = self.chunk_remainder[:chunk_size_bytes]
                self.chunk_remainder = self.chunk_remainder[chunk_size_bytes:]

                audio_np = np.frombuffer(chunk_bytes, dtype=np.int16).astype(np.float32) / 32768.0

                c16 = self.limiter.process(audio_np).flatten()
                self.history_q.append(c16)

                if self.vad is not None:
                    # ONNXVAD is not async, but we can run it in event loop as it's fast enough
                    # for real-time chunks (32ms).
                    res = self.vad(c16)
                    if not self.state_speech and res == "start":
                        self.state_speech = True
                        hq_list = list(self.history_q)
                        self.speech_buffer.extend(hq_list[:-1])
                        self.speech_buffer.append(c16)
                    elif self.state_speech:
                        self.speech_buffer.append(c16)
                        if res == "end":
                            self.state_speech = False
                            self.history_q.clear()
                else:
                    self.speech_buffer.append(c16)

    def start_recording(self, source="mic"):
        if self.is_recording:
            return

        self.is_recording = True
        self.audio_chunks = []

        session = get_vad_session()
        self.vad = ONNXVAD(session) if session is not None else None
        self.limiter = TrueRMSLimiter(target_rms=TARGET_RMS, max_gain=MAX_GAIN)
        self.speech_buffer = []
        self.history_q = deque(maxlen=15)
        self.state_speech = False
        self.chunk_remainder = b""

        title = get_string("rec_mic_title") if source == "mic" else get_string("rec_sys_title")
        msg = get_string("rec_mic_msg") if source == "mic" else get_string("rec_sys_msg")

        notif_id = send_notification(title, msg)
        if notif_id:
            with open(ID_FILE, "w") as f:
                f.write(notif_id)

        # Play start sound
        play_sound("/usr/share/sounds/freedesktop/stereo/service-login.oga")

        input_device = "default" if source == "mic" else "@DEFAULT_SINK@.monitor"

        ffmpeg_cmd = [
            "ffmpeg", "-nostdin", "-v", "quiet", "-f", "pulse", "-i", input_device,
            "-metadata", "title=groq_audio", "-ac", "1", "-ar", "16000",
            "-f", "s16le", "-"
        ]

        asyncio.create_task(self._start_ffmpeg(ffmpeg_cmd))

        # Start the 10-minute timeout task
        self.timeout_task = asyncio.create_task(self._recording_timeout())

    async def _start_ffmpeg(self, ffmpeg_cmd):
        self.process = await asyncio.create_subprocess_exec(
            *ffmpeg_cmd,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL
        )
        self.read_task = asyncio.create_task(self._read_stdout())

    def stop_recording(self):
        if not self.is_recording:
            return

        self.is_recording = False

        if self.timeout_task and not self.timeout_task.done():
            self.timeout_task.cancel()

        # Play end sound
        play_sound("/usr/share/sounds/freedesktop/stereo/service-logout.oga")

        repl_id = None
        if os.path.exists(ID_FILE):
            with open(ID_FILE, "r") as f:
                repl_id = f.read().strip()

        notif_id = send_notification(get_string("processing_title"), get_string("processing_msg"), repl_id)
        if notif_id:
            with open(ID_FILE, "w") as f:
                f.write(notif_id)

        if self.process:
            self.process.terminate()

        # We schedule processing in background
        self.processing_task = asyncio.create_task(self._process_collected_audio())

    async def _process_collected_audio(self):
        repl_id = None
        if os.path.exists(ID_FILE):
            with open(ID_FILE, "r") as f:
                repl_id = f.read().strip()

        try:
            if self.read_task:
                await self.read_task

            if self.process:
                if self.process.returncode is None:
                    try:
                        self.process.kill()
                    except ProcessLookupError:
                        pass
                await self.process.wait()

            if not self.speech_buffer:
                logging.info("No speech detected during recording.")
                text = ""
            else:
                audio_np = np.concatenate(self.speech_buffer)
                if len(audio_np) < RATE_WHISPER * 0.5:
                    logging.info("Speech audio too short.")
                    text = ""
                else:
                    wav_file_data = _make_wav(audio_np, 16000)
                    ogg_data = wav_to_ogg(wav_file_data)
                    text, _ = await process_ready_audio(ogg_data)

            # Close the "Processing..." notification explicitly so the next one pops up as a fresh notification
            if repl_id:
                close_notification(repl_id)

            last_id = None
            if "продолжение следует" in text.lower():
                # Technically an error/silence state, play error or silence sound
                play_sound("/usr/share/sounds/freedesktop/stereo/message.oga")
                last_id = send_notification(get_string("silence_title"), get_string("silence_msg"))
            elif text and text != "null":
                if os.path.exists(FIX_FLAG_FILE):
                    text = await fix_text_with_llm(text)
                    os.remove(FIX_FLAG_FILE)

                subprocess.run(["wl-copy"], input=text, text=True)
                clean_text = " ".join(text.splitlines())[:40]
                if len(text) > 40:
                    clean_text += "..."

                # Play success transcription sound
                play_sound("/usr/share/sounds/freedesktop/stereo/message-new-instant.oga")

                last_id = send_notification(get_string("success_title"), clean_text)
            else:
                # Play error sound
                play_sound("/usr/share/sounds/freedesktop/stereo/message.oga")
                last_id = send_notification(get_string("error_title"), get_string("error_msg"))

            # Let the notification stay for 4 seconds, then close it
            await asyncio.sleep(4)
            if last_id:
                close_notification(last_id)

            if os.path.exists(ID_FILE):
                os.remove(ID_FILE)
        except Exception as e:
            logging.error(f"Error during audio processing: {e}")
            if repl_id:
                close_notification(repl_id)
            play_sound("/usr/share/sounds/freedesktop/stereo/message.oga")
            last_id = send_notification(get_string("sys_error_title"), str(e))
            await asyncio.sleep(4)
            if last_id:
                close_notification(last_id)
            if os.path.exists(ID_FILE):
                os.remove(ID_FILE)
        finally:
            if os.path.exists(FIX_FLAG_FILE):
                os.remove(FIX_FLAG_FILE)

    def toggle(self):
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording(source="mic")

    def toggle_sys(self):
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording(source="sys")

async def main():
    # Write PID so we can signal this process
    with open(PID_FILE, "w") as f:
        f.write(str(os.getpid()))

    daemon = AudioDaemon()
    loop = asyncio.get_running_loop()

    # Bind toggle to SIGUSR1
    loop.add_signal_handler(signal.SIGUSR1, daemon.toggle)
    loop.add_signal_handler(signal.SIGUSR2, daemon.toggle_sys)

    stop_event = asyncio.Event()

    # Graceful shutdown on SIGINT/SIGTERM
    def shutdown():
        stop_event.set()

    loop.add_signal_handler(signal.SIGINT, shutdown)
    loop.add_signal_handler(signal.SIGTERM, shutdown)

    logging.info(f"Daemon started with PID {os.getpid()}. Waiting for SIGUSR1 to toggle recording.")

    try:
        await stop_event.wait()
    finally:
        if daemon.is_recording:
            daemon.stop_recording()
            if daemon.processing_task:
                await daemon.processing_task
        if os.path.exists(PID_FILE):
            os.remove(PID_FILE)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
