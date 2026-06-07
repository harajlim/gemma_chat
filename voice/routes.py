"""
FastAPI routes for the realtime voice pipeline — a faithful port of the
`real_time_voice` server endpoints so the voice experience is unchanged:

    POST /api/converse_stream   voice turn  (audio in -> NDJSON: transcript, reply_text, audio…, done)
    POST /api/converse_text     typed turn  (text in  -> same NDJSON, AI replies in audio)
    POST /api/speak             TTS playground (text -> streamed audio)
    GET  /api/voice/config      current brain/voice/system-prompt + options
    POST /api/voice/config      live-update brain/voice/engine/system-prompt
    GET  /api/voice/health      ollama + tts readiness
    POST /api/voice/reset       clear the spoken conversation history

NDJSON event protocol (one JSON object per line), identical to the voice app:
    {"type":"transcript","text": "..."}
    {"type":"reply_text","text": "..."}
    {"type":"audio","sr": 24000, "pcm": "<base64 int16 mono>"}
    {"type":"done","first_audio": 1.23}
    {"type":"error","message": "..."}
"""

import base64
import json as _json
import os
import time

from fastapi import APIRouter, UploadFile, File, Form, Body
from fastapi.responses import JSONResponse, StreamingResponse

from . import engine

router = APIRouter()


@router.get("/api/voice/health")
def voice_health():
    return {
        "ok": True,
        "model": engine._state["model"],
        "ollama": engine.ollama_ok(),
        "tts": engine.tts_label(),
    }


@router.post("/api/voice/reset")
def voice_reset():
    engine.reset_history()
    return {"ok": True}


@router.get("/api/voice/config")
def voice_get_config():
    return {
        "model": engine._state["model"],
        "models": engine.list_chat_models(),
        "voice": engine.current_voice(),
        "voices": engine.available_voices(),
        "system_prompt": engine.get_system_prompt(),
        "default_system_prompt": engine.SYSTEM_PROMPT,
    }


@router.post("/api/voice/config")
def voice_set_config(cfg: dict = Body(...)):
    state = engine.set_config(
        model=cfg.get("model"),
        voice=cfg.get("voice"),
        system_prompt=cfg.get("system_prompt"),
    )
    return {"ok": True, "model": state["model"], "voice": engine.current_voice()}


def _turn_stream(user_text_or_none, raw_audio=None, suffix=".webm"):
    """Shared NDJSON generator for voice + typed turns. If raw_audio is given we
    transcribe it first; otherwise user_text_or_none is used directly."""
    t0 = time.time()
    try:
        if raw_audio is not None:
            wav_path = engine.to_wav16k(raw_audio, suffix)
            try:
                user_text = engine.transcribe(wav_path)
            finally:
                if os.path.exists(wav_path):
                    os.unlink(wav_path)
        else:
            user_text = (user_text_or_none or "").strip()

        yield _json.dumps({"type": "transcript", "text": user_text}) + "\n"
        if not user_text:
            yield _json.dumps({"type": "done"}) + "\n"
            return

        reply = engine.clean_for_tts(engine.chat(user_text))
        yield _json.dumps({"type": "reply_text",
                           "text": engine.strip_emotion_tags(reply)}) + "\n"

        first_audio = None
        for chunk, sr in engine.tts_stream(reply):
            if not chunk:
                continue
            if first_audio is None:
                first_audio = time.time() - t0
            yield _json.dumps({
                "type": "audio", "sr": sr,
                "pcm": base64.b64encode(bytes(chunk)).decode("ascii"),
            }) + "\n"

        yield _json.dumps({"type": "done", "first_audio": round(first_audio or 0, 2)}) + "\n"
        print(f"[voice] total {time.time()-t0:.2f}s | first_audio {first_audio} "
              f"-> you: {user_text!r}")
    except Exception as e:  # noqa: BLE001
        # Always close the stream with a terminal frame so the client never hangs.
        print(f"[voice] error: {e}")
        yield _json.dumps({"type": "error", "message": str(e)}) + "\n"
        yield _json.dumps({"type": "done"}) + "\n"


@router.post("/api/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    """Speech -> text only (Whisper). Used by the 'talk -> text reply' combo,
    which feeds the transcript into the rich WS chat instead of the voice TTS."""
    raw = await audio.read()
    suffix = os.path.splitext(audio.filename or "")[1] or ".webm"
    wav_path = engine.to_wav16k(raw, suffix)
    try:
        text = engine.transcribe(wav_path)
    finally:
        if os.path.exists(wav_path):
            os.unlink(wav_path)
    return JSONResponse({"text": text})


@router.post("/api/converse_stream")
async def converse_stream(audio: UploadFile = File(...)):
    """Streaming voice turn: STT -> LLM -> sentence-streamed TTS audio."""
    raw = await audio.read()
    suffix = os.path.splitext(audio.filename or "")[1] or ".webm"
    return StreamingResponse(_turn_stream(None, raw_audio=raw, suffix=suffix),
                             media_type="application/x-ndjson")


@router.post("/api/converse_text")
async def converse_text(text: str = Form(...)):
    """Typed turn: skip STT, reuse chat + TTS streaming; AI replies in audio."""
    return StreamingResponse(_turn_stream(text),
                             media_type="application/x-ndjson")


@router.post("/api/speak")
async def speak(text: str = Form(...), voice: str = Form(None)):
    """TTS playground: synthesize arbitrary text (with emotion tags), stream audio."""
    def gen():
        try:
            for chunk, sr in engine.tts_stream(text, voice):
                if not chunk:
                    continue
                yield _json.dumps({
                    "type": "audio", "sr": sr,
                    "pcm": base64.b64encode(bytes(chunk)).decode("ascii"),
                }) + "\n"
            yield _json.dumps({"type": "done"}) + "\n"
        except Exception as e:  # noqa: BLE001
            print(f"[speak] error: {e}")
            yield _json.dumps({"type": "error", "message": str(e)}) + "\n"
            yield _json.dumps({"type": "done"}) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson")
