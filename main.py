import os
import sys
import signal

PID_FILE = "/tmp/groq_audio_daemon.pid"

# Quick toggle path that doesn't require any third-party libraries
if len(sys.argv) > 1 and sys.argv[1] == "toggle":
    try:
        with open(PID_FILE, "r") as f:
            pid = int(f.read().strip())
        os.kill(pid, signal.SIGUSR1)
        print(f"Sent toggle signal to daemon (PID: {pid})")
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
import json
from collections import deque

import aiohttp
import numpy as np
import av
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

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


def _extract_audio_sync(audio_bytes: bytes) -> np.ndarray | None:
    """Synchronous helper to extract audio from audio bytes."""
    try:
        with av.open(io.BytesIO(audio_bytes)) as container:
            stream = container.streams.audio[0]
            resampler = av.AudioResampler(
                format="s16", layout="mono", rate=16000
            )

            pcm_chunks = []
            for frame in container.decode(stream):
                for r_frame in resampler.resample(frame):
                    pcm_chunks.append(r_frame.to_ndarray())

            if not pcm_chunks:
                raise RuntimeError("No frames decoded")

            raw_audio = (
                np.concatenate(pcm_chunks, axis=1).flatten().astype(np.float32)
            )
            return raw_audio
    except Exception as e:
        logging.error(f"Failed to decode audio bytes: {e}")
        return None

def _apply_vad_and_limit_sync(raw_audio: np.ndarray) -> bytes | None:
    """Synchronous CPU-bound function to apply TrueRMS and VAD, returning OGG bytes."""
    session = get_vad_session()
    vad = ONNXVAD(session) if session is not None else None
    limiter = TrueRMSLimiter(target_rms=TARGET_RMS, max_gain=MAX_GAIN)

    roll_ch = raw_audio / 32768.0

    buffer = []
    history_q: deque[np.ndarray] = deque(maxlen=15)
    state_speech = False

    while len(roll_ch) >= CHUNK_VAD_16K:
        c16 = roll_ch[:CHUNK_VAD_16K].copy().flatten()
        roll_ch = roll_ch[CHUNK_VAD_16K:]

        c16 = limiter.process(c16).flatten()
        history_q.append(c16)

        if vad is not None:
            res = vad(c16)
            if not state_speech and res == "start":
                state_speech = True
                hq_list = list(history_q)
                buffer.extend(hq_list[:-1])
                buffer.append(c16)
            elif state_speech:
                buffer.append(c16)
                if res == "end":
                    state_speech = False
                    history_q.clear()
        else:
            buffer.append(c16)

    if not buffer:
        logging.info("No speech detected in input.")
        return None

    audio_np = np.concatenate(buffer)
    if len(audio_np) < RATE_WHISPER * 0.5:
        logging.info("Audio too short after VAD processing.")
        return None

    wav_file_data = _make_wav(audio_np, 16000)
    return wav_to_ogg(wav_file_data)


async def process_audio_bytes(audio_bytes: bytes) -> tuple[str, bytes]:
    """
    Reads input audio bytes, applies denoise, VAD, gain,
    outputs (final text, processed wav bytes) from Groq Whisper.
    """
    if not audio_bytes or len(audio_bytes) < 100:
        logging.info("Audio too small or empty.")
        return "", b""

    raw_audio = await asyncio.to_thread(_extract_audio_sync, audio_bytes)
    if raw_audio is None:
        return "", b""

    file_data = await asyncio.to_thread(_apply_vad_and_limit_sync, raw_audio)
    if file_data is None:
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
            async with session_http.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
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

    async def _recording_timeout(self):
        # Wait for 10 minutes (600 seconds)
        await asyncio.sleep(600)
        if self.is_recording:
            logging.info("10-minute recording limit reached. Automatically stopping.")
            self.stop_recording()

    async def _read_stdout(self):
        while self.process and not self.process.stdout.at_eof():
            chunk = await self.process.stdout.read(4096)
            if chunk:
                self.audio_chunks.append(chunk)

    def start_recording(self):
        if self.is_recording:
            return

        self.is_recording = True
        self.audio_chunks = []

        notif_id = send_notification("🎙️ Запись", "Говорите...")
        if notif_id:
            with open(ID_FILE, "w") as f:
                f.write(notif_id)

        # Play start sound
        subprocess.Popen(
            ["paplay", "/usr/share/sounds/freedesktop/stereo/service-login.oga"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        ffmpeg_cmd = [
            "ffmpeg", "-nostdin", "-v", "quiet", "-f", "pulse", "-i", "default",
            "-metadata", "title=groq_audio", "-ac", "1", "-ar", "16000",
            "-c:a", "libopus", "-f", "ogg", "-"
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
        subprocess.Popen(
            ["paplay", "/usr/share/sounds/freedesktop/stereo/service-logout.oga"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        repl_id = None
        if os.path.exists(ID_FILE):
            with open(ID_FILE, "r") as f:
                repl_id = f.read().strip()

        notif_id = send_notification("🛑 Обработка...", "Ждем ответ от API", repl_id)
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

            audio_bytes = b"".join(self.audio_chunks)

            text, _ = await process_audio_bytes(audio_bytes)

            last_id = None
            if "продолжение следует" in text.lower():
                last_id = send_notification("🔇 Тишина", "Голос не обнаружен", repl_id)
            elif text and text != "null":
                subprocess.run(["wl-copy"], input=text, text=True)
                clean_text = " ".join(text.splitlines())[:40]
                if len(text) > 40:
                    clean_text += "..."
                last_id = send_notification("✅ Запись обработана", clean_text, repl_id)
            else:
                last_id = send_notification("❌ Ошибка", "Не удалось распознать текст", repl_id)

            # Let the notification stay for 4 seconds, then close it
            await asyncio.sleep(4)
            if last_id:
                close_notification(last_id)

            if os.path.exists(ID_FILE):
                os.remove(ID_FILE)
        except Exception as e:
            logging.error(f"Error during audio processing: {e}")
            last_id = send_notification("❌ Системная Ошибка", str(e), repl_id)
            await asyncio.sleep(4)
            if last_id:
                close_notification(last_id)
            if os.path.exists(ID_FILE):
                os.remove(ID_FILE)

    def toggle(self):
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()

async def main():
    # Write PID so we can signal this process
    with open(PID_FILE, "w") as f:
        f.write(str(os.getpid()))

    daemon = AudioDaemon()
    loop = asyncio.get_running_loop()

    # Bind toggle to SIGUSR1
    loop.add_signal_handler(signal.SIGUSR1, daemon.toggle)

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
