import os
import asyncio
import io
import struct
import aiohttp
import numpy as np
import av
import logging
from collections import deque
import subprocess
import signal
import sys
import json

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
        return "", b""

    audio_np = np.concatenate(buffer)
    if len(audio_np) < RATE_WHISPER * 0.5:
        logging.info("Audio too short after VAD processing.")
        return "", b""

    file_data = _make_wav(audio_np, 16000)

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
        filename="processed_audio.wav",
        content_type="audio/wav",
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

PID_FILE = "/tmp/groq_audio.pid"
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

async def main():
    # If already running, toggle off
    if os.path.exists(PID_FILE):
        try:
            with open(PID_FILE, "r") as f:
                pid = int(f.read().strip())

            # Send SIGINT to the running process
            os.kill(pid, signal.SIGINT)

            # Update notification to Processing
            repl_id = None
            if os.path.exists(ID_FILE):
                with open(ID_FILE, "r") as f:
                    repl_id = f.read().strip()

            notif_id = send_notification("🛑 Обработка...", "Ждем ответ от API", repl_id)
            if notif_id:
                with open(ID_FILE, "w") as f:
                    f.write(notif_id)

            return # Exit this instance
        except ProcessLookupError:
            # Process was not running, cleanup and continue
            os.remove(PID_FILE)
        except Exception as e:
            logging.error(f"Error checking PID: {e}")
            pass

    # Start new recording instance
    with open(PID_FILE, "w") as f:
        f.write(str(os.getpid()))

    notif_id = send_notification("🎙️ Запись", "Говорите...")
    if notif_id:
        with open(ID_FILE, "w") as f:
            f.write(notif_id)

    ffmpeg_cmd = [
        "ffmpeg", "-v", "quiet", "-f", "pulse", "-i", "default",
        "-metadata", "title=groq_audio", "-ac", "1", "-ar", "16000",
        "-c:a", "libopus", "-f", "ogg", "-"
    ]

    process = await asyncio.create_subprocess_exec(
        *ffmpeg_cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL
    )

    stop_recording = asyncio.Event()

    def signal_handler(sig, frame):
        stop_recording.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    audio_chunks = []

    async def read_stdout():
        while not process.stdout.at_eof():
            chunk = await process.stdout.read(4096)
            if chunk:
                audio_chunks.append(chunk)

    read_task = asyncio.create_task(read_stdout())

    try:
        while not stop_recording.is_set():
            await asyncio.sleep(0.1)

        process.terminate()
        await read_task
    finally:
        if process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
        await process.wait()

        # Audio collected, we can remove the PID file to allow a new recording to start
        if os.path.exists(PID_FILE):
            os.remove(PID_FILE)

    audio_bytes = b"".join(audio_chunks)

    # Audio collected, process it
    repl_id = None
    if os.path.exists(ID_FILE):
        with open(ID_FILE, "r") as f:
            repl_id = f.read().strip()

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

    await asyncio.sleep(4)
    if last_id:
        close_notification(last_id)

    if os.path.exists(ID_FILE):
        os.remove(ID_FILE)

if __name__ == "__main__":
    asyncio.run(main())
