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
import time
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


async def process_ready_audio(file_data: bytes, prompt: str = "") -> tuple[str, bytes]:
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

    if prompt:
        data.add_field("prompt", prompt)

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

        # Chunking/Streaming state
        self.transcription_queue = None
        self.worker_task = None

        self.current_notif_id = None

        self.last_final_transcription = ""
        self.last_final_transcription_time = 0.0

    def _show_notification(self, title, msg):
        new_id = send_notification(title, msg, self.current_notif_id)
        if new_id:
            self.current_notif_id = new_id

    def _close_notification(self):
        if self.current_notif_id:
            close_notification(self.current_notif_id)
            self.current_notif_id = None

    async def _recording_timeout(self):
        # Wait for 10 minutes (600 seconds)
        await asyncio.sleep(600)
        if self.is_recording:
            logging.info("10-minute recording limit reached. Automatically stopping.")
            self.stop_recording()

    async def _read_stdout(self, process, limiter, vad, history_q, speech_buffer, transcription_queue):
        chunk_size_bytes = CHUNK_VAD_16K * 2
        chunk_remainder = b""
        state_speech = False

        while process and not process.stdout.at_eof():
            chunk = await process.stdout.read(4096)
            if not chunk:
                continue

            chunk_remainder += chunk

            while len(chunk_remainder) >= chunk_size_bytes:
                chunk_bytes = chunk_remainder[:chunk_size_bytes]
                chunk_remainder = chunk_remainder[chunk_size_bytes:]

                audio_np = np.frombuffer(chunk_bytes, dtype=np.int16).astype(np.float32) / 32768.0

                c16 = limiter.process(audio_np).flatten()
                history_q.append(c16)

                if vad is not None:
                    # ONNXVAD is not async, but we can run it in event loop as it's fast enough
                    # for real-time chunks (32ms).
                    res = vad(c16)
                    if not state_speech and res == "start":
                        state_speech = True
                        hq_list = list(history_q)
                        speech_buffer.extend(hq_list[:-1])
                        speech_buffer.append(c16)
                    elif state_speech:
                        speech_buffer.append(c16)
                        if res == "end":
                            state_speech = False
                            history_q.clear()

                            # Check if we should dispatch a chunk (>= 10 seconds of speech)
                            buffer_len_seconds = (len(speech_buffer) * CHUNK_VAD_16K) / RATE_WHISPER
                            if buffer_len_seconds >= 10.0:
                                chunk_to_send = list(speech_buffer)
                                speech_buffer.clear()
                                logging.debug(f"Dispatching chunk of {buffer_len_seconds:.2f}s")
                                transcription_queue.put_nowait(chunk_to_send)
                else:
                    speech_buffer.append(c16)

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

        self.transcription_queue = asyncio.Queue()
        self.full_transcription = []

        title = get_string("rec_mic_title") if source == "mic" else get_string("rec_sys_title")
        msg = get_string("rec_mic_msg") if source == "mic" else get_string("rec_sys_msg")

        input_device = "default" if source == "mic" else "@DEFAULT_SINK@.monitor"

        ffmpeg_cmd = [
            "ffmpeg", "-nostdin", "-v", "quiet", "-f", "pulse", "-i", input_device,
            "-metadata", "title=groq_audio", "-ac", "1", "-ar", "16000",
            "-f", "s16le", "-"
        ]

        # Start ffmpeg capture task first so recording begins immediately
        asyncio.create_task(self._start_ffmpeg(ffmpeg_cmd, self.transcription_queue, self.full_transcription))

        # Play start sound
        play_sound("/usr/share/sounds/freedesktop/stereo/service-login.oga")

        self._show_notification(title, msg)

        # Start the 10-minute timeout task
        self.timeout_task = asyncio.create_task(self._recording_timeout())

        initial_prompt = ""
        keep_context = os.environ.get("KEEP_CONTEXT_BETWEEN_RECORDINGS", "true").lower() == "true"
        if keep_context and self.last_final_transcription:
            time_since_last = time.time() - self.last_final_transcription_time
            if time_since_last < 15.0:
                words = self.last_final_transcription.split()
                initial_prompt = " ".join(words[-15:]) if len(words) > 15 else self.last_final_transcription

        # Start transcription worker
        self.worker_task = asyncio.create_task(
            self._transcription_worker(self.transcription_queue, self.full_transcription, prompt=initial_prompt)
        )

    async def _transcription_worker(self, queue, full_transcription_list, prompt=""):
        while True:
            chunk = await queue.get()
            if chunk is None:  # EOF marker
                queue.task_done()
                break

            chunk_start = time.perf_counter()
            audio_np = np.concatenate(chunk)
            if len(audio_np) >= RATE_WHISPER * 0.5:
                wav_file_data = _make_wav(audio_np, 16000)
                ogg_data = wav_to_ogg(wav_file_data)

                # Send to API
                text, _ = await process_ready_audio(ogg_data, prompt=prompt)

                if text and text != "null" and "продолжение следует" not in text.lower():
                    full_transcription_list.append(text)

                    # Update prompt with the last ~10-15 words
                    words = text.split()
                    prompt = " ".join(words[-15:]) if len(words) > 15 else text

            elapsed = time.perf_counter() - chunk_start
            logging.info(f"Chunk processing (including API) took {elapsed:.2f}s")

            queue.task_done()

    async def _start_ffmpeg(self, ffmpeg_cmd, queue, full_transcription):
        process = await asyncio.create_subprocess_exec(
            *ffmpeg_cmd,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL
        )

        # In case stop was pressed before this task started
        if not self.is_recording:
            process.terminate()
            return

        self.process = process

        self.read_task = asyncio.create_task(self._read_stdout(
            process, self.limiter, self.vad, self.history_q,
            self.speech_buffer, queue
        ))

    def stop_recording(self):
        if not self.is_recording:
            return

        self.is_recording = False

        if self.timeout_task and not self.timeout_task.done():
            self.timeout_task.cancel()

        # Play end sound
        play_sound("/usr/share/sounds/freedesktop/stereo/service-logout.oga")

        self._show_notification(get_string("processing_title"), get_string("processing_msg"))

        # Capture the current state and reset instance variables so immediate new recordings
        # get fresh state.
        old_process = self.process
        old_read_task = self.read_task
        old_worker_task = self.worker_task
        old_queue = self.transcription_queue
        old_speech_buffer = self.speech_buffer
        old_full_transcription = self.full_transcription

        self.process = None
        self.read_task = None
        self.worker_task = None
        self.transcription_queue = None
        self.speech_buffer = []
        self.full_transcription = []

        if old_process:
            old_process.terminate()

        # We schedule processing in background
        self.processing_task = asyncio.create_task(self._process_collected_audio(
            old_process, old_read_task, old_worker_task, old_queue, old_speech_buffer, old_full_transcription
        ))

    async def _process_collected_audio(self, process, read_task, worker_task, queue, speech_buffer, full_transcription):
        try:
            finish_start_time = time.perf_counter()
            if read_task:
                await read_task

            if process:
                if process.returncode is None:
                    try:
                        process.kill()
                    except ProcessLookupError:
                        pass
                await process.wait()

            # Flush remaining audio
            if speech_buffer and queue is not None:
                queue.put_nowait(list(speech_buffer))
                speech_buffer.clear()

            # Send EOF and wait for worker to finish
            if queue is not None:
                queue.put_nowait(None)

            if worker_task is not None:
                await worker_task

            finish_elapsed = time.perf_counter() - finish_start_time
            logging.info(f"Finalizing transcription queue took {finish_elapsed:.2f}s")

            # Close the "Processing..." notification explicitly so the next one pops up as a fresh notification
            self._close_notification()

            text = " ".join(full_transcription).strip()

            if text and text != "null":
                if os.path.exists(FIX_FLAG_FILE):
                    text = await fix_text_with_llm(text)
                    os.remove(FIX_FLAG_FILE)

                # 1. Copy to clipboard immediately
                subprocess.run(["wl-copy"], input=text, text=True)

                # 2. Play sound
                play_sound("/usr/share/sounds/freedesktop/stereo/message-new-instant.oga")

                # 3. Format and show notification
                clean_text = " ".join(text.splitlines())[:40]
                if len(text) > 40:
                    clean_text += "..."

                self._show_notification(get_string("success_title"), clean_text)

                self.last_final_transcription = text
                self.last_final_transcription_time = time.time()
            else:
                # Play error sound/silence sound
                play_sound("/usr/share/sounds/freedesktop/stereo/message.oga")
                if not full_transcription:
                    self._show_notification(get_string("silence_title"), get_string("silence_msg"))
                else:
                    self._show_notification(get_string("error_title"), get_string("error_msg"))

            # Let the notification stay for 4 seconds, then close it
            await asyncio.sleep(4)
            self._close_notification()

        except Exception as e:
            logging.error(f"Error during audio processing: {e}")
            self._close_notification()
            play_sound("/usr/share/sounds/freedesktop/stereo/message.oga")
            self._show_notification(get_string("sys_error_title"), str(e))
            await asyncio.sleep(4)
            self._close_notification()
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
    is_debug = os.environ.get("DEBUG", "false").lower() == "true"
    logging.basicConfig(
        level=logging.DEBUG if is_debug else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    asyncio.run(main())
