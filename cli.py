#!/usr/bin/env python3
"""vowit CLI — прямой доступ к Groq API (Whisper + LLM) из терминала.

Работает поверх ядра проекта vowit: переиспользует тракт обработки звука
(Silero VAD, TrueRMSLimiter), упаковку в WAV/OGG и LLM-фикс из main.py.

Примеры:
    ~/vowit/.venv/bin/python ~/vowit/cli.py transcribe "note.m4a" -o note.txt
    ~/vowit/.venv/bin/python ~/vowit/cli.py transcribe "note.m4a" --no-vad --model whisper-large-v3-turbo
    ~/vowit/.venv/bin/python ~/vowit/cli.py translate "note.m4a"
    ~/vowit/.venv/bin/python ~/vowit/cli.py fix note.txt --llm-model openai/gpt-oss-120b
    ~/vowit/.venv/bin/python ~/vowit/cli.py models
"""

import argparse
import asyncio
import os
import subprocess
import sys
from collections import deque

import aiohttp
import numpy as np
from dotenv import load_dotenv

script_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(script_dir, ".env"))

from main import (  # noqa: E402
    CHUNK_VAD_16K,
    MAX_GAIN,
    RATE_WHISPER,
    TARGET_RMS,
    TrueRMSLimiter,
    _make_wav,
    fix_text_with_llm,
    get_vad_session,
    wav_to_ogg,
)

TRANS_URL = os.environ.get(
    "GROQ_API_URL", "https://api.groq.com/openai/v1/audio/transcriptions"
)
TRANS_URL_EN = "https://api.groq.com/openai/v1/audio/translations"

DEFAULT_WHISPER = os.environ.get("WHISPER_MODEL", "whisper-large-v3")
DEFAULT_LANG = os.environ.get("BOT_LANGUAGE", "ru")
DEFAULT_LLM = os.environ.get("LLM_MODEL", "openai/gpt-oss-120b")

WHISPER_MODELS = ("whisper-large-v3", "whisper-large-v3-turbo")


# --------------------------------------------------------------------------
# Audio processing (reuses the vowit signal chain)
# --------------------------------------------------------------------------

def load_audio(path: str) -> np.ndarray:
    """Decodes any audio/video file into 16kHz mono float32 via ffmpeg."""
    cmd = [
        "ffmpeg", "-nostdin", "-v", "quiet", "-i", path,
        "-ac", "1", "-ar", "16000", "-f", "s16le", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.decode(errors='replace')}")
    return np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0


def split_speech(audio: np.ndarray, chunk_seconds: float, use_vad: bool):
    """Splits raw audio into speech batches.

    With VAD: replicates the daemon's chain (TrueRMSLimiter -> Silero VAD),
    dropping silence and keeping chunks of at least `chunk_seconds` of speech.
    Without VAD: plain equal slices of `chunk_seconds`.
    """
    chunk_size = CHUNK_VAD_16K

    if not use_vad:
        sec_size = int(RATE_WHISPER * chunk_seconds)
        batches = [
            audio[i:i + sec_size]
            for i in range(0, len(audio), sec_size)
            if len(audio[i:i + sec_size]) >= RATE_WHISPER * 0.5
        ]
        return batches or [audio]

    session = get_vad_session()
    if session is None:
        print("[warn] Silero VAD недоступен, режем без него", file=sys.stderr)
        return split_speech(audio, chunk_seconds, use_vad=False)

    from main import ONNXVAD
    vad = ONNXVAD(session)
    limiter = TrueRMSLimiter(target_rms=TARGET_RMS, max_gain=MAX_GAIN)

    batches, buffer = [], []
    history = deque(maxlen=15)
    state_speech = False

    for i in range(0, len(audio) - chunk_size + 1, chunk_size):
        c16 = limiter.process(audio[i:i + chunk_size]).flatten()
        history.append(c16)

        res = vad(c16)
        if not state_speech and res == "start":
            state_speech = True
            buffer.extend(list(history)[:-1])
            buffer.append(c16)
        elif state_speech:
            buffer.append(c16)
            if res == "end":
                state_speech = False
                history.clear()
            if (len(buffer) * chunk_size) / RATE_WHISPER >= chunk_seconds:
                batches.append(np.concatenate(buffer))
                buffer = []

    if buffer:
        batches.append(np.concatenate(buffer))
    return batches or [audio]


def batch_to_ogg(batch: np.ndarray) -> bytes:
    """Packs a speech batch into OGG OPUS (the format used by the daemon)."""
    return wav_to_ogg(_make_wav(batch, RATE_WHISPER))


# --------------------------------------------------------------------------
# Groq API helpers
# --------------------------------------------------------------------------

def _audio_form(ogg: bytes, model: str, language: str | None, prompt: str) -> aiohttp.FormData:
    fd = aiohttp.FormData()
    fd.add_field("model", model)
    fd.add_field("temperature", "0")
    if language:
        fd.add_field("language", language)
    if prompt:
        fd.add_field("prompt", prompt)
    fd.add_field("file", ogg, filename="vowit_cli.ogg", content_type="audio/ogg")
    return fd


async def _api_audio(form: aiohttp.FormData, url: str, retries: int = 10) -> str:
    """POSTs a multipart form to Groq, auto-retrying on rate limits."""
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY не задан в .env")

    headers = {"Authorization": f"Bearer {api_key}"}
    for attempt in range(retries):
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, data=form) as resp:
                if resp.status == 200:
                    return (await resp.json()).get("text", "").strip()
                err = await resp.text()
                if resp.status == 429:
                    print(f"[retry {attempt + 1}/{retries}] 429 rate limit, ждём...",
                          file=sys.stderr)
                    await asyncio.sleep(3 + attempt * 4)
                    continue
                raise RuntimeError(f"Groq API {resp.status}: {err}")
    raise RuntimeError("rate limit: попытки исчерпаны")


async def _transcribe_batch(ogg: bytes, model: str, language: str, prompt: str) -> str:
    return await _api_audio(_audio_form(ogg, model, language, prompt), TRANS_URL)


async def _translate_batch(ogg: bytes, model: str, prompt: str) -> str:
    return await _api_audio(_audio_form(ogg, model, "en", prompt), TRANS_URL_EN)


def _last_words(text: str, n: int = 15) -> str:
    words = text.split()
    return " ".join(words[-n:]) if len(words) > n else text


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------

async def _process_audio_files(paths, model, language, chunk, use_vad, output, translate=False):
    """Shared pipeline for transcribe/translate."""
    results = []
    for path in paths:
        audio = load_audio(path)
        batches = split_speech(audio, chunk, use_vad)
        speech_secs = sum(len(b) / RATE_WHISPER for b in batches)
        tag = "речь" if use_vad else "аудио"
        print(f"{os.path.basename(path)}: {speech_secs:.1f}s {tag}, "
              f"{len(batches)} батч(ев)", file=sys.stderr)

        prompt, parts = "", []
        for n, batch in enumerate(batches, 1):
            ogg = batch_to_ogg(batch)
            text = (await _translate_batch(ogg, model, prompt)) if translate \
                else (await _transcribe_batch(ogg, model, language, prompt))
            print(f"  [{n}/{len(batches)}] {len(text)} симв.", file=sys.stderr)
            if text and text != "null":
                parts.append(text)
                prompt = _last_words(text)

        result = " ".join(parts).strip()
        if output:
            with open(output, "w", encoding="utf-8") as f:
                f.write(result)
            print(f"  сохранено: {output}", file=sys.stderr)
        else:
            print(f"--- {os.path.basename(path)} ---")
            print(result)
            print()
        results.append(result)
    return results


async def cmd_transcribe(args):
    if not args.files:
        print("Нет файлов.", file=sys.stderr)
        return 1
    await _process_audio_files(
        args.files, args.model, args.language, args.chunk,
        args.vad, args.output, translate=False,
    )
    return 0


async def cmd_translate(args):
    if not args.files:
        print("Нет файлов.", file=sys.stderr)
        return 1
    await _process_audio_files(
        args.files, args.model, "en", args.chunk,
        args.vad, args.output, translate=True,
    )
    return 0


async def cmd_fix(args):
    for path in args.files:
        text = sys.stdin.read() if path == "-" else open(path, encoding="utf-8").read()
        text = text.strip()
        if not text:
            continue

        if args.llm_model:
            os.environ["LLM_MODEL"] = args.llm_model
        if args.llm_url:
            os.environ["LLM_BASE_URL"] = args.llm_url

        fixed = await fix_text_with_llm(text)
        if fixed == "network_error":
            print("Ошибка сети при вызове LLM.", file=sys.stderr)
            fixed = text

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(fixed)
            print(f"  сохранено: {args.output}", file=sys.stderr)
        else:
            print(f"--- {os.path.basename(path) if path != '-' else 'stdin'} ---")
            print(fixed)
            print()
    return 0


def cmd_models(_args):
    print("Whisper (транскрибация/перевод):")
    for m in WHISPER_MODELS:
        print(f"  {m}")
    print("LLM (fix, грамматика):")
    print(f"  {DEFAULT_LLM}")
    print(f"\nТекущие настройки (.env):")
    print(f"  WHISPER_MODEL = {DEFAULT_WHISPER}")
    print(f"  BOT_LANGUAGE  = {DEFAULT_LANG}")
    return 0


# --------------------------------------------------------------------------
# CLI entrypoint
# --------------------------------------------------------------------------

def _add_audio_common(p):
    p.add_argument("files", nargs="+", help="аудио/видео файлы (m4a, ogg, wav, mp3...)")
    p.add_argument("-m", "--model", default=DEFAULT_WHISPER,
                   help=f"модель Whisper (default: {DEFAULT_WHISPER})")
    p.add_argument("-o", "--output", help="сохранить текст в файл вместо stdout")
    p.add_argument("--chunk", type=float, default=60.0,
                   help="мин. размер батча в секундах (default: 60)")
    p.add_argument("--vad", action="store_true", default=True,
                   help="тракт VAD: убрать тишину, нормализовать громкость (default)")
    p.add_argument("--no-vad", dest="vad", action="store_false",
                   help="без VAD, резать аудио ровными кусками")


def build_parser():
    parser = argparse.ArgumentParser(
        prog="vowit-cli",
        description="Прямой доступ к Groq API (Whisper + LLM) через терминал.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_tr = sub.add_parser("transcribe", help="аудио -> текст (распознавание речи)")
    _add_audio_common(p_tr)
    p_tr.add_argument("-l", "--language", default=DEFAULT_LANG,
                      help=f"язык ISO-639-1 (default: {DEFAULT_LANG})")
    p_tr.set_defaults(func=cmd_transcribe)

    p_tl = sub.add_parser("translate", help="аудио -> английский текст (перевод)")
    _add_audio_common(p_tl)
    p_tl.set_defaults(func=cmd_translate)

    p_fx = sub.add_parser("fix", help="исправить грамматику текста через LLM")
    p_fx.add_argument("files", nargs="+", help="текстовые файлы (или '-' для stdin)")
    p_fx.add_argument("-o", "--output", help="сохранить результат в файл")
    p_fx.add_argument("--llm-model", help=f"модель LLM (default: {DEFAULT_LLM})")
    p_fx.add_argument("--llm-url", help="переопределить base URL для LLM")
    p_fx.set_defaults(func=cmd_fix)

    p_md = sub.add_parser("models", help="показать доступные модели и настройки")
    p_md.set_defaults(func=cmd_models)

    return parser


async def main():
    args = build_parser().parse_args()
    try:
        res = args.func(args)
        if asyncio.iscoroutine(res):
            rc = await res
        else:
            rc = res
    except (RuntimeError, OSError, ValueError) as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        return 1
    return rc or 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
