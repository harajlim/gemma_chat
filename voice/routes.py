"""
FastAPI routes for the realtime voice pipeline.

Voice turns now run the SAME tools as the text chat (web_search + detect_objects).
When a tool fires the voice first SPEAKS a short announcement ("Let me search the
web for…"), then runs the tool, then speaks the grounded answer — all in Orpheus.

NDJSON event protocol (one JSON object per line):
    {"type":"transcript","text": "..."}
    {"type":"reply_text","text": "..."}            # a spoken segment (announcement or answer)
    {"type":"audio","sr": 24000, "pcm": "<base64 int16 mono>"}
    {"type":"web_search","query":..,"results":[..],"search_time_s":..}
    {"type":"detection","image_url":..,"target":..,"count":..,"bboxes":[..],"detection_time_s":..}
    {"type":"done","first_audio": 1.23}
    {"type":"error","message": "..."}
"""

import base64
import json as _json
import os
import time

from fastapi import APIRouter, UploadFile, File, Form, Body
from fastapi.responses import JSONResponse, StreamingResponse
from ollama import chat as ollama_chat

import tools
from . import engine

router = APIRouter()

MAX_TOOL_STEPS = 5
# Ceiling only — the spoken system prompt keeps replies short; this just stops a
# runaway. Generous enough not to truncate a tool call or a grounded sentence.
_NUM_PREDICT = 256


@router.get("/api/voice/health")
def voice_health():
    return {"ok": True, "model": engine._state["model"], "ollama": engine.ollama_ok(), "tts": engine.tts_label()}


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
    state = engine.set_config(model=cfg.get("model"), voice=cfg.get("voice"), system_prompt=cfg.get("system_prompt"))
    return {"ok": True, "model": state["model"], "voice": engine.current_voice()}


_TOOL_HINT = (
    "You can use tools: web_search for current/online facts, and detect_objects to find things "
    "in an image the user attached. Call a tool when it would help, then reply in one short spoken "
    "sentence. Do not mention tool names or JSON; just answer naturally. When you use web_search, "
    "answer ONLY from the returned results; if the specific value isn't in them, say you couldn't "
    "find it rather than guessing. For weather questions you may web_search even without a city — "
    "the result uses the local area — so prefer searching over asking which city."
)


def _voice_reply(user_text: str, image_ids):
    """Tool-aware voice brain. Yields semantic events for the caller to render:
       ("speak", text)  — a spoken segment (announcement or final answer)
       ("card", dict)   — a web_search / detection card to show in the thread
    Commits exactly the user turn + final answer to the short voice history."""
    model = engine._state["model"]
    valid_imgs = [i for i in (image_ids or []) if tools.valid_id(i)]

    with engine._lock:
        base = list(engine._history)

    # Extra guidance is folded into the LEADING system message rather than added
    # as separate system turns: some chat templates (e.g. qwen) require the system
    # message to be first and ONLY first, and raise otherwise.
    extra = [_TOOL_HINT]
    if valid_imgs:
        lst = ", ".join(f"image (id: {i})" for i in valid_imgs)
        extra.append(f"The user attached: {lst}. When calling detect_objects, use the image_id.")

    if base and base[0].get("role") == "system":
        head = {"role": "system", "content": (base[0]["content"] + " " + " ".join(extra)).strip()}
        messages = [head] + base[1:]
    else:
        messages = [{"role": "system", "content": " ".join(extra)}] + base

    user_msg = {"role": "user", "content": user_text}
    media = []
    for i in valid_imgs:
        for p in tools.UPLOAD_DIR.glob(f"{i}.*"):
            media.append(str(p)); break
    if media:
        user_msg["images"] = media

    messages.append(user_msg)
    final_text = None

    for _ in range(MAX_TOOL_STEPS):
        resp = ollama_chat(model=model, messages=messages, tools=tools.TOOLS,
                           think=False, options={"num_predict": _NUM_PREDICT})
        tcs = resp.message.tool_calls
        if not tcs:
            final_text = resp.message.content or ""
            break

        messages.append({
            "role": "assistant", "content": resp.message.content or "",
            "tool_calls": [{"function": {"name": tc.function.name, "arguments": tc.function.arguments}} for tc in tcs],
        })
        for tc in tcs:
            name, args = tc.function.name, tc.function.arguments
            yield ("speak", tools.announce(name, args))   # spoken "what I'm doing"

            if name == "web_search":
                res = tools.run_web_search(args.get("query", ""))
                yield ("card", {"type": "web_search", "query": res["query"], "results": res["results"], "search_time_s": res["search_time_s"]})
                messages.append({"role": "tool", "content": _json.dumps({"query": res["query"], "results": res["results"]})})
            elif name == "detect_objects":
                iid = args.get("image_id", "")
                img = tools.resolve_image_bytes(iid) or (tools.resolve_image_bytes(valid_imgs[-1]) if valid_imgs else None)
                if img is None:
                    messages.append({"role": "tool", "content": "No image available for detection."})
                else:
                    det = tools.run_detection(img, args.get("target", "objects"), model)
                    yield ("card", {"type": "detection", "image_url": det["image_url"], "target": det["target"],
                                    "count": det["count"], "bboxes": det["bboxes"], "detection_time_s": det["detection_time_s"]})
                    messages.append({"role": "tool", "content": _json.dumps({"count": det["count"], "target": det["target"]})})
            else:
                messages.append({"role": "tool", "content": f"Unknown tool: {name}"})
    else:
        # hit the step cap — force a final spoken answer
        resp = ollama_chat(model=model, messages=messages, think=False, options={"num_predict": _NUM_PREDICT})
        final_text = resp.message.content or ""

    if not final_text:
        final_text = "Sorry, I couldn't find an answer to that."

    with engine._lock:
        engine._history.append({"role": "user", "content": user_text})
        engine._history.append({"role": "assistant", "content": final_text})

    yield ("speak", final_text)


def _turn_stream(user_text_or_none, raw_audio=None, suffix=".webm", image_ids=None):
    """Shared NDJSON generator for voice + typed turns."""
    t0 = time.time()
    first_audio = [None]
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

        for ev in _voice_reply(user_text, image_ids):
            if ev[0] == "speak":
                cleaned = engine.clean_for_tts(ev[1])
                if not cleaned:
                    continue
                yield _json.dumps({"type": "reply_text", "text": engine.strip_emotion_tags(cleaned)}) + "\n"
                for chunk, sr in engine.tts_stream(cleaned):
                    if not chunk:
                        continue
                    if first_audio[0] is None:
                        first_audio[0] = round(time.time() - t0, 2)
                    yield _json.dumps({"type": "audio", "sr": sr, "pcm": base64.b64encode(bytes(chunk)).decode("ascii")}) + "\n"
            elif ev[0] == "card":
                yield _json.dumps(ev[1]) + "\n"

        yield _json.dumps({"type": "done", "first_audio": first_audio[0] or 0}) + "\n"
        print(f"[voice] total {time.time()-t0:.2f}s | first_audio {first_audio[0]} -> you: {user_text!r}")
    except Exception as e:  # noqa: BLE001
        print(f"[voice] error: {e}")
        yield _json.dumps({"type": "error", "message": str(e)}) + "\n"
        yield _json.dumps({"type": "done"}) + "\n"


def _parse_ids(s):
    return [i for i in (s or "").split(",") if i.strip()]


@router.post("/api/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    """Speech -> text only (Whisper). Used by the 'talk -> text reply' combo."""
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
async def converse_stream(audio: UploadFile = File(...), image_ids: str = Form("")):
    """Streaming voice turn: STT -> tool-aware brain -> sentence-streamed TTS."""
    raw = await audio.read()
    suffix = os.path.splitext(audio.filename or "")[1] or ".webm"
    return StreamingResponse(_turn_stream(None, raw_audio=raw, suffix=suffix, image_ids=_parse_ids(image_ids)),
                             media_type="application/x-ndjson")


@router.post("/api/converse_text")
async def converse_text(text: str = Form(...), image_ids: str = Form("")):
    """Typed turn: skip STT, reuse the tool-aware brain + TTS; reply in audio."""
    return StreamingResponse(_turn_stream(text, image_ids=_parse_ids(image_ids)),
                             media_type="application/x-ndjson")


@router.post("/api/speak")
async def speak(text: str = Form(...), voice: str = Form(None)):
    """TTS playground: synthesize arbitrary text (with emotion tags), stream audio."""
    def gen():
        try:
            for chunk, sr in engine.tts_stream(text, voice):
                if not chunk:
                    continue
                yield _json.dumps({"type": "audio", "sr": sr, "pcm": base64.b64encode(bytes(chunk)).decode("ascii")}) + "\n"
            yield _json.dumps({"type": "done"}) + "\n"
        except Exception as e:  # noqa: BLE001
            yield _json.dumps({"type": "error", "message": str(e)}) + "\n"
            yield _json.dumps({"type": "done"}) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson")
