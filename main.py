import os
import sys
import signal
import glob

PID_FILE = "/tmp/groq_audio_daemon.pid"

FIX_FLAG_FILE = "/tmp/groq_audio_fixon.flag"

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

script_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(script_dir, ".env")
load_dotenv(env_path)

SOUNDS_DIR = os.path.join(script_dir, "sounds")

# Auto-detect Wayland display socket if not set
if "WAYLAND_DISPLAY" not in os.environ:
    sockets = glob.glob(f"/run/user/{os.getuid()}/wayland-[0-9]*")
    if sockets:
        os.environ["WAYLAND_DISPLAY"] = os.path.basename(sockets[0])

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
        "sys_error_title": "❌ System Error",
        "net_error_title": "📡 Network Error",
        "net_error_msg": "Failed to connect to API"
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
        "sys_error_title": "❌ Системная Ошибка",
        "net_error_title": "📡 Ошибка сети",
        "net_error_msg": "Нет доступа к сети / API"
    }
}

def get_string(key):
    lang = UI_LANGUAGE if UI_LANGUAGE in STRINGS else "en"
    return STRINGS[lang].get(key, STRINGS["en"].get(key, key))

TARGET_RMS = float(os.environ.get("TARGET_RMS", "0.06"))
MAX_GAIN = float(os.environ.get("MAX_GAIN", "2.0"))
RATE_WHISPER = 16000
CHUNK_VAD_16K = 512
CHUNK_SPEECH_SECONDS = float(os.environ.get("CHUNK_SPEECH_SECONDS", "15.0"))
VAD_THRESHOLD = float(os.environ.get("VAD_THRESHOLD", "0.5"))
VAD_MIN_SILENCE_MS = int(os.environ.get("VAD_MIN_SILENCE_MS", "200"))
MULTI_PRESS_WINDOW = float(os.environ.get("MULTI_PRESS_WINDOW", "0.6"))
STOP_COOLDOWN = float(os.environ.get("STOP_COOLDOWN", "1.0"))
RECORDING_TIMEOUT = int(os.environ.get("RECORDING_TIMEOUT", "600"))
NOTIFICATION_DURATION = int(os.environ.get("NOTIFICATION_DURATION", "4"))
FFMPEG_GRACE = float(os.environ.get("FFMPEG_GRACE", "0.3"))
PASTE_ENTER_DELAY = float(os.environ.get("PASTE_ENTER_DELAY", "0.2"))
FIX_SKIP_SHORT = os.environ.get("FIX_SKIP_SHORT", "true").lower() == "true"
FIX_MIN_CHARS = int(os.environ.get("FIX_MIN_CHARS", "50"))

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
        self, session, threshold=VAD_THRESHOLD, min_silence_ms=VAD_MIN_SILENCE_MS, chunk_ms=32
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


LLM_MODEL_DEFAULT = "openai/gpt-oss-120b"
LLM_FALLBACK_DELAY = float(os.environ.get("LLM_FALLBACK_DELAY", "0.8"))


def _llm_model_rotation():
    """Returns up to 5 models to try for grammar fixing, primary first."""
    primary = os.environ.get("LLM_MODEL", LLM_MODEL_DEFAULT)
    fallbacks = [
        m.strip() for m in os.environ.get(
            "LLM_FALLBACK_MODELS",
            "qwen/qwen3.6-27b,llama-3.3-70b-versatile,openai/gpt-oss-20b"
        ).split(",")
        if m.strip()
    ]
    models = [primary] + [m for m in fallbacks if m != primary]
    return models[:5]


async def fix_text_with_llm(text: str) -> str:
    """Uses an LLM to slightly fix grammar and add paragraphs to the transcribed text.

    Tries the primary model first; if it errors out or returns an empty response,
    rotates through fallback models (up to 5 attempts total) until one succeeds.
    Returns "network_error" if all models fail.
    """
    groq_api_key = os.environ.get("GROQ_API_KEY", "")
    if not groq_api_key:
        return text

    headers = {
        "Authorization": f"Bearer {groq_api_key}",
        "Content-Type": "application/json"
    }

    # Allow overriding the base URL in case they use OpenRouter or another provider for the LLM fix
    llm_base_url = os.environ.get("LLM_BASE_URL", "https://api.groq.com/openai/v1/chat/completions")

    # Backup (Variant 1) — компактный, в стиле старого промпта:
    # system_prompt = (
    #     "Your task is to slightly fix grammar and divide the text into paragraphs. "
    #     "The text is a transcribed voice memo, so it may contain miswordings or "
    #     "phonetic errors. If the speaker asks for formatting inside the message "
    #     "(e.g., \"summarize the beginning in 20 words\" or \"split into paragraphs\"), "
    #     "follow that request, but remove the request itself from the output. "
    #     "Keep the original language of the text; do not translate it unless the "
    #     "speaker explicitly asks to. Do not use the em dash (—) symbol. "
    #     "Example: raw: \"meet at cafe tomorrow noon bring laptops\" -> fixed: "
    #     "\"Meet at the cafe tomorrow at noon. Bring laptops.\" "
    #     "Make minimal changes. Do not remake the text fully. Respond ONLY with the fixed text."
    # )
    system_prompt = (
        "Your task is to clean up a transcribed voice memo: fix grammar and punctuation, "
        "and divide the text into paragraphs. Remember it is raw speech — expect phonetic "
        "errors, dropped words, and missing punctuation. Make minimal changes; do not rewrite "
        "the message. Keep the original language of the text; do not translate it into another "
        "language unless the speaker explicitly asks to. Do not use the em dash (—) symbol; "
        "use commas, colons, or other punctuation instead. "
        "If the speaker explicitly asks for something inside the message (e.g., "
        "\"summarize the beginning in 20 words\", \"format as a list\", \"add headings\"), "
        "honor that request, then delete the request itself from the result so only the "
        "actual content remains. "
        "Example:\nRaw:  \"meet at cafe tomorrow noon bring laptops\"\n"
        "Fixed: \"Meet at the cafe tomorrow at noon. Bring laptops.\"\n"
        "Respond ONLY with the fixed text."
    )

    for model in _llm_model_rotation():
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            "temperature": 0.2
        }

        try:
            start_time = time.perf_counter()
            async with aiohttp.ClientSession() as session_http:
                async with session_http.post(
                    llm_base_url,
                    headers=headers,
                    json=payload,
                ) as resp:
                    if resp.status == 429:
                        err = await resp.text()
                        logging.warning(f"LLM {model} rate limited (429): {err}")
                        await asyncio.sleep(3)
                        continue
                    if resp.status != 200:
                        err = await resp.text()
                        logging.warning(f"LLM {model} API Error {resp.status}: {err}")
                        await asyncio.sleep(LLM_FALLBACK_DELAY)
                        continue
                    json_resp = await resp.json()
                    fixed_text = json_resp.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                    if not fixed_text:
                        logging.warning(f"LLM {model} returned empty response")
                        await asyncio.sleep(LLM_FALLBACK_DELAY)
                        continue
                    elapsed = time.perf_counter() - start_time
                    logging.info(f"LLM grammar correction ({model}) took {elapsed:.2f}s")
                    return fixed_text
        except aiohttp.ClientError as e:
            logging.warning(f"Network error calling LLM API with {model}: {e}")
        except Exception as e:
            logging.warning(f"Error calling LLM API with {model}: {e}")
            await asyncio.sleep(LLM_FALLBACK_DELAY)

    return "network_error"


async def process_ready_audio(file_data: bytes, prompt: str = "") -> tuple[str, bytes]:
    """
    Sends the pre-processed audio bytes to Groq Whisper.
    """
    if not file_data:
        return "", b""

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

    for attempt in range(3):
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
        except aiohttp.ClientError as e:
            logging.error(f"Network error calling Groq Whisper API (attempt {attempt + 1}/3): {e}")
            if attempt < 2:
                await asyncio.sleep(1)
        except Exception as e:
            logging.error(f"Error calling Groq API: {e}")
            return "", b""

    return "network_error", b""

SOUND_VOLUME = os.environ.get("SOUND_VOLUME", "0.8")
SOUND_SPEED = os.environ.get("SOUND_SPEED", "2.0")

def play_sound(sound_path):
    subprocess.run(["pkill", "-x", "ffplay"], stderr=subprocess.DEVNULL)
    subprocess.Popen(
        ["ffplay", "-nodisp", "-autoexit", "-af", f"volume={SOUND_VOLUME},atempo={SOUND_SPEED}", sound_path],
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
        notif_id = result.stdout.strip()
        logging.debug(f"Notification sent: '{title}' id={notif_id}")
        return notif_id
    except subprocess.CalledProcessError as e:
        logging.warning(f"notify-send failed: {e.stderr}")
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
        self.read_task = None
        self.processing_task = None
        self.timeout_task = None
        self._last_toggle = 0.0
        self._recording_start_time = 0.0
        self._presses = 0
        self._last_press_time = 0.0
        self._output_mode = None  # "clipboard", "paste", None
        self._stop_cooldown = 0.0
        self._stop_pending = False

        # Real-time processing state
        self.vad = None
        self.limiter = None
        self.speech_buffer = []
        self.history_q = None

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
        await asyncio.sleep(RECORDING_TIMEOUT)
        if self.is_recording:
            logging.info(f"{RECORDING_TIMEOUT}s recording limit reached. Automatically stopping.")
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
                            if buffer_len_seconds >= CHUNK_SPEECH_SECONDS:
                                chunk_to_send = list(speech_buffer)
                                speech_buffer.clear()
                                logging.debug(f"Dispatching chunk of {buffer_len_seconds:.2f}s")
                                transcription_queue.put_nowait(chunk_to_send)
                else:
                    speech_buffer.append(c16)

    def start_recording(self, source="mic", startup_sound=True):
        if self.is_recording:
            return

        self.is_recording = True
        self.audio_chunks = []

        session = get_vad_session()
        self.vad = ONNXVAD(session, threshold=VAD_THRESHOLD) if session is not None else None
        self.limiter = TrueRMSLimiter(target_rms=TARGET_RMS, max_gain=MAX_GAIN)
        self.speech_buffer = []
        self.history_q = deque(maxlen=15)

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

        asyncio.create_task(self._start_ffmpeg(ffmpeg_cmd, self.transcription_queue, self.full_transcription))

        if startup_sound:
            play_sound(os.path.join(SOUNDS_DIR, "service-login.oga"))

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

                if text == "network_error":
                    full_transcription_list.append("network_error")
                    # Stop parsing further chunks if we have a critical network failure
                    queue.task_done()
                    break
                elif text and text != "null" and "продолжение следует" not in text.lower():
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
        play_sound(os.path.join(SOUNDS_DIR, "service-logout.oga"))

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

        self.processing_task = asyncio.create_task(self._process_collected_audio(
            old_process, old_read_task, old_worker_task, old_queue, old_speech_buffer, old_full_transcription
        ))

    async def _process_collected_audio(
        self, process, read_task, worker_task, queue, speech_buffer, full_transcription
    ):
        try:
            finish_start_time = time.perf_counter()

            if process and process.returncode is None:
                await asyncio.sleep(FFMPEG_GRACE)
                try:
                    process.terminate()
                except ProcessLookupError:
                    pass

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

            if "network_error" in full_transcription:
                play_sound(os.path.join(SOUNDS_DIR, "message.oga"))
                self._show_notification(get_string("net_error_title"), get_string("net_error_msg"))

                # Filter out the error string and copy whatever text successfully transcribed beforehand
                clean_transcription = " ".join([t for t in full_transcription if t != "network_error"]).strip()
                if clean_transcription and clean_transcription != "null":
                    subprocess.run(["wl-copy"], input=clean_transcription, text=True)
            elif text and text != "null":
                llm_failed = False
                fix_enabled = os.path.exists(FIX_FLAG_FILE) or os.environ.get(
                    "FIX_GRAMMAR_BY_DEFAULT", "false"
                ).lower() == "true"

                if fix_enabled:
                    if FIX_SKIP_SHORT and len(text) < FIX_MIN_CHARS:
                        logging.info(f"Text too short ({len(text)} chars < {FIX_MIN_CHARS}), skipping LLM fix")
                    else:
                        text = await fix_text_with_llm(text)
                    if os.path.exists(FIX_FLAG_FILE):
                        os.remove(FIX_FLAG_FILE)
                    if text == "network_error":
                        llm_failed = True
                        play_sound(os.path.join(SOUNDS_DIR, "message.oga"))
                        self._show_notification(get_string("net_error_title"), get_string("net_error_msg"))
                        text = " ".join(full_transcription).strip() # fallback to un-fixed text to at least copy it

                if text != "network_error" and not llm_failed:
                    # Copy to clipboard
                    subprocess.run(["wl-copy"], input=text, text=True)

                    if self._output_mode == "paste":
                        subprocess.run(["wtype", "-M", "ctrl", "-M", "shift", "-k", "v"])
                        await asyncio.sleep(PASTE_ENTER_DELAY)
                        subprocess.run(["wtype", "-k", "Return"])

                    # Play sound
                    play_sound(os.path.join(SOUNDS_DIR, "message-new-instant.oga"))

                    # Format and show notification
                    clean_text = " ".join(text.splitlines())[:40]
                    if len(text) > 40:
                        clean_text += "..."

                    self._show_notification(get_string("success_title"), clean_text)

                    self.last_final_transcription = text
                    self.last_final_transcription_time = time.time()
                elif llm_failed:
                    # Just silently copy the text if LLM failed, we already showed the error notification
                    subprocess.run(["wl-copy"], input=text, text=True)
            else:
                # Play error sound/silence sound
                play_sound(os.path.join(SOUNDS_DIR, "message.oga"))
                if not full_transcription:
                    self._show_notification(get_string("silence_title"), get_string("silence_msg"))
                else:
                    self._show_notification(get_string("error_title"), get_string("error_msg"))

            await asyncio.sleep(NOTIFICATION_DURATION)
            self._close_notification()

        except Exception as e:
            logging.error(f"Error during audio processing: {e}")
            self._close_notification()
            play_sound(os.path.join(SOUNDS_DIR, "message.oga"))
            self._show_notification(get_string("sys_error_title"), str(e))
            await asyncio.sleep(NOTIFICATION_DURATION)
            self._close_notification()
        finally:
            if os.path.exists(FIX_FLAG_FILE):
                os.remove(FIX_FLAG_FILE)

    def toggle(self):
        now = time.time()
        if now - self._last_toggle < 0.125:
            return
        self._last_toggle = now

        if not self.is_recording and self._stop_pending:
            self._stop_pending = False
            self._output_mode = "paste"
            self._stop_cooldown = now
            return

        if self._stop_cooldown > 0 and now - self._stop_cooldown < STOP_COOLDOWN:
            return

        if not self.is_recording:
            self._recording_start_time = now
            self._last_press_time = now
            self.start_recording(source="mic")
            self._presses = 1
        else:
            dt = now - self._recording_start_time
            if dt < MULTI_PRESS_WINDOW:
                if now - self._last_press_time < 0.125:
                    return
                self._last_press_time = now
                self._presses += 1
                if self._presses >= 3:
                    self._cancel_recording()
                    self._stop_cooldown = now
                    asyncio.create_task(self._start_sys_with_delay())
            else:
                self._output_mode = "clipboard"
                self._stop_pending = True
                self._stop_cooldown = now
                self.stop_recording()
                asyncio.create_task(self._stop_timeout())

    def toggle_sys(self):
        now = time.time()
        if now - self._last_toggle < 0.5:
            return
        self._last_toggle = now
        if self.is_recording:
            self._stop_cooldown = now
            self.stop_recording()
        else:
            self.start_recording(source="sys")

    async def _stop_timeout(self):
        await asyncio.sleep(MULTI_PRESS_WINDOW)
        self._stop_pending = False

    async def _start_sys_with_delay(self):
        await asyncio.sleep(0.5)
        self._recording_start_time = time.time()
        self.start_recording(source="sys", startup_sound=False)

    def _cancel_recording(self):
        if not self.is_recording:
            return
        self.is_recording = False
        if self.timeout_task and not self.timeout_task.done():
            self.timeout_task.cancel()
        if self.process:
            try:
                self.process.terminate()
            except ProcessLookupError:
                pass
        if self.read_task and not self.read_task.done():
            self.read_task.cancel()
        if self.worker_task and not self.worker_task.done():
            self.worker_task.cancel()
        if self.transcription_queue:
            self.transcription_queue = None
        self.process = None
        self.read_task = None
        self.worker_task = None
        self.transcription_queue = None
        self.speech_buffer = []
        self.full_transcription = []
        self._close_notification()

async def main():
    if os.path.exists(PID_FILE):
        try:
            with open(PID_FILE, "r") as f:
                old_pid = int(f.read().strip())
            os.kill(old_pid, 0)
            logging.error(f"Daemon already running with PID {old_pid}. Exiting.")
            sys.exit(1)
        except (ProcessLookupError, ValueError):
            pass
        os.remove(PID_FILE)

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
