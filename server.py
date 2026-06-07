"""
Gemma 4 Chat + Voice — one cohesive local app.

Two capabilities, one server, one conversation surface:

  • Rich TEXT chat (WebSocket /ws/chat) — streaming tokens, markdown, image upload
    & multimodal, tool calling (web_search + detect_objects), live throughput stats.
    [unchanged from the original testing_gemma app]

  • Realtime VOICE (REST NDJSON, see voice/routes.py) — hands-free VAD mic loop:
    Whisper STT -> Ollama brain -> Orpheus TTS, streamed back & played
    gaplessly. [behaviour-identical to the real_time_voice app]

The "brain" (default gemma4:12b) is shared: picking a model in the UI points both
the text chat and the voice loop at the same Ollama model.
"""

import asyncio
import base64
import json
import re
import subprocess
import threading
import time
import uuid
from pathlib import Path

# Uploaded media ids are uuid4().hex (32 lowercase hex). Validating against this
# before any filesystem glob neutralises path-traversal via client-sent ids.
_ID_RE = re.compile(r"^[0-9a-f]{32}$")

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from ollama import chat as ollama_chat

import tools
from voice import engine as voice_engine
from voice.routes import router as voice_router

app = FastAPI()

# --- State ---
conversations: dict[str, list[dict]] = {}  # session_id -> messages
uploaded_images: dict[str, bytes] = {}  # image_id -> raw bytes
session_images: dict[str, list[dict]] = {}  # session_id -> [{id, filename, index}]

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
DETECTION_DIR = Path("detection")
DETECTION_DIR.mkdir(exist_ok=True)

# --- Models ---------------------------------------------------------------
# The brain picker lists EVERY usable Ollama chat model on this machine (just
# like the realtime_voice app did) — gemma4:*, qwen-unc, whatever you've pulled —
# excluding the TTS model. `vision` gates image input / detection; `audio` gates
# Gemma's native audio-attachment understanding (edge models only). The realtime
# voice loop uses Whisper for STT, so it works with any brain regardless of flags.
DEFAULT_MODEL = "gemma4:12b"
_KNOWN_CAPS = {
    "gemma4:12b": {"vision": True, "audio": False},
    "gemma4:e4b": {"vision": True, "audio": True},
    "gemma4:e2b": {"vision": True, "audio": True},
    "gemma4:26b": {"vision": True, "audio": False},
}
session_models: dict[str, str] = {}  # session_id -> model name


def list_model_info() -> dict:
    """Every Ollama chat model -> capability flags. Unknown models default to
    text-only (vision/audio False) but are still selectable as the brain."""
    names = voice_engine.list_chat_models()
    if DEFAULT_MODEL not in names:
        names = [DEFAULT_MODEL, *names]
    return {n: dict(_KNOWN_CAPS.get(n, {"vision": False, "audio": False})) for n in names}


# Keep the voice brain in sync with the chat default at boot.
voice_engine.set_config(model=DEFAULT_MODEL)

# Tools (web_search + detect_objects) are shared with the voice path — see tools.py.
TOOLS = tools.TOOLS


# --- Routes ---

@app.get("/models")
async def list_models():
    return {"models": list_model_info(), "default": DEFAULT_MODEL}


@app.get("/")
async def index():
    return FileResponse("static/index.html")


@app.get("/tts")
async def tts_page():
    return FileResponse("static/tts.html")


@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """Upload an image and return an ID for referencing it in chat."""
    data = await file.read()
    image_id = uuid.uuid4().hex
    # Save to disk
    ext = Path(file.filename or "image.png").suffix or ".png"
    path = UPLOAD_DIR / f"{image_id}{ext}"
    path.write_bytes(data)
    # Also keep raw bytes in memory for detection
    uploaded_images[image_id] = data
    return {"image_id": image_id, "filename": file.filename}


@app.post("/upload_audio")
async def upload_audio(file: UploadFile = File(...)):
    """Upload an audio file, convert to WAV, and return an ID."""
    data = await file.read()
    audio_id = uuid.uuid4().hex
    ext = Path(file.filename or "audio.webm").suffix or ".webm"

    # Save original
    orig_path = UPLOAD_DIR / f"{audio_id}_orig{ext}"
    orig_path.write_bytes(data)

    # Convert to WAV (16kHz mono) for reliable Ollama compatibility
    wav_path = UPLOAD_DIR / f"{audio_id}.wav"
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(orig_path), "-ar", "16000", "-ac", "1", str(wav_path)],
            capture_output=True, check=True, timeout=30,
        )
        orig_path.unlink(missing_ok=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        # If ffmpeg fails, keep original and hope for the best
        wav_path = UPLOAD_DIR / f"{audio_id}{ext}"
        if not wav_path.exists():
            orig_path.rename(wav_path)

    return {"audio_id": audio_id, "filename": file.filename}


@app.get("/images/{image_id}")
async def get_image(image_id: str):
    """Serve an uploaded or generated image."""
    # Check uploads
    for p in UPLOAD_DIR.glob(f"{image_id}.*"):
        return FileResponse(p)
    # Check detection results
    for p in DETECTION_DIR.glob(f"{image_id}.*"):
        return FileResponse(p)
    return Response(status_code=404)


@app.get("/audio/{audio_id}")
async def get_audio(audio_id: str):
    """Serve an uploaded audio file."""
    wav = UPLOAD_DIR / f"{audio_id}.wav"
    if wav.exists():
        return FileResponse(wav, media_type="audio/wav")
    for p in UPLOAD_DIR.glob(f"{audio_id}.*"):
        if "_orig" not in p.name:
            return FileResponse(p)
    return Response(status_code=404)


# Realtime voice pipeline (STT -> brain -> TTS). Behaviour-identical to the
# original real_time_voice app; see voice/routes.py + voice/engine.py.
app.include_router(voice_router)


# --- WebSocket Chat ---

@app.websocket("/ws/chat")
async def websocket_chat(ws: WebSocket):
    await ws.accept()
    session_id = None

    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)

            # Init or resume session
            if session_id is None:
                session_id = msg.get("session_id") or uuid.uuid4().hex
                if session_id not in conversations:
                    conversations[session_id] = []
                if session_id not in session_images:
                    session_images[session_id] = []
                if session_id not in session_models:
                    session_models[session_id] = DEFAULT_MODEL
                print(f"[session {session_id[:8]}] init, model={session_models[session_id]}")
                await ws.send_text(json.dumps({"type": "session", "session_id": session_id}))

            # Handle model switch — keep the voice brain in sync so the whole app
            # uses ONE brain.
            if msg.get("type") == "set_model":
                new_model = msg.get("model", DEFAULT_MODEL)
                if new_model in list_model_info():
                    session_models[session_id] = new_model
                    voice_engine.set_config(model=new_model)
                print(f"[session {session_id[:8]}] model switched to {session_models[session_id]}")
                await ws.send_text(json.dumps({"type": "model_set", "model": session_models[session_id]}))
                continue

            text = msg.get("text", "")
            # Only accept well-formed ids (uuid4 hex) — never let a client-sent
            # id reach a filesystem glob (path-traversal guard).
            image_ids = [i for i in msg.get("image_ids", []) if isinstance(i, str) and _ID_RE.match(i)]
            audio_ids = [a for a in msg.get("audio_ids", []) if isinstance(a, str) and _ID_RE.match(a)]

            # Skip messages with no content (e.g. session init)
            if not text and not image_ids and not audio_ids:
                continue

            # Track images for this session (so the model knows what's available)
            if image_ids:
                for iid in image_ids:
                    idx = len(session_images[session_id]) + 1
                    session_images[session_id].append({"id": iid, "index": idx})

            # Build user message for ollama
            has_audio = bool(audio_ids)

            # If only audio and no text, add a default prompt
            if has_audio and not text:
                text = "Listen to this audio and respond to what is being said."

            user_msg: dict = {"role": "user", "content": text}

            # Collect media: images as file paths, audio as base64
            media = []
            if image_ids:
                for iid in image_ids:
                    for p in UPLOAD_DIR.glob(f"{iid}.*"):
                        media.append(str(p))
                        break
            if audio_ids:
                for aid in audio_ids:
                    wav = UPLOAD_DIR / f"{aid}.wav"
                    audio_path = wav if wav.exists() else next(UPLOAD_DIR.glob(f"{aid}.*"), None)
                    if audio_path and audio_path.exists():
                        audio_b64 = base64.b64encode(audio_path.read_bytes()).decode()
                        media.append(audio_b64)
            if media:
                user_msg["images"] = media

            conversations[session_id].append(user_msg)

            # Audio messages bypass tool-calling (tools mode breaks audio understanding).
            # Guarantee a terminal frame even if generation throws (e.g. Ollama
            # error, OOM, or a non-JSON detection reply) — otherwise the client
            # never sees `done` and its turn state (and the voice orb) get wedged.
            try:
                await _generate_response(ws, session_id, use_tools=not has_audio, has_audio=has_audio)
            except WebSocketDisconnect:
                raise
            except Exception as e:
                print(f"[session {session_id[:8]}] generation error: {e}")
                try:
                    await ws.send_text(json.dumps({"type": "error", "text": f"Generation failed: {e}"}))
                    await ws.send_text(json.dumps({"type": "done"}))
                except Exception:
                    pass

    except WebSocketDisconnect:
        pass


async def _stream_response(ws: WebSocket, session_id: str, model: str,
                           messages: list[dict], options: dict | None = None):
    """Pure streaming call — no tools. Used for audio and the post-tool follow-up."""
    import queue

    q: queue.Queue = queue.Queue()
    start = time.perf_counter()

    def _run():
        """Run blocking ollama stream in a thread, push chunks to queue."""
        try:
            kwargs = {"model": model, "messages": messages, "stream": True}
            if options:
                kwargs["options"] = options
            for chunk in ollama_chat(**kwargs):
                q.put(("token", chunk.message.content))
            q.put(("done", None))
        except Exception as e:
            q.put(("error", str(e)))

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    first_token_time = None
    token_count = 0
    full_text_parts = []

    while True:
        # Poll the queue, yielding to the event loop between checks
        try:
            msg_type, data = q.get(timeout=0.05)
        except queue.Empty:
            await asyncio.sleep(0.01)
            continue

        if msg_type == "token":
            if first_token_time is None:
                first_token_time = time.perf_counter()
            if data:
                full_text_parts.append(data)
                token_count += 1
                await ws.send_text(json.dumps({"type": "token", "text": data}))
        elif msg_type == "done":
            break
        elif msg_type == "error":
            await ws.send_text(json.dumps({"type": "error", "text": data}))
            break

    thread.join(timeout=5)

    end = time.perf_counter()
    full_text = "".join(full_text_parts)
    conversations[session_id].append({"role": "assistant", "content": full_text})

    gen_time = end - (first_token_time or end)
    stats = {
        "type": "stats",
        "model": model,
        "total_time_s": round(end - start, 2),
        "ttft_s": round((first_token_time or end) - start, 2),
        "stream_time_s": round(gen_time, 2),
        "approx_tokens": token_count,
        "tokens_per_sec": round(token_count / max(gen_time, 0.001), 1),
    }
    await ws.send_text(json.dumps(stats))
    await ws.send_text(json.dumps({"type": "done"}))


async def _execute_tool_call(ws: WebSocket, session_id: str, model: str, tc) -> None:
    """Execute a single tool call and append the result to conversation history."""
    fn_name = tc.function.name
    fn_args = tc.function.arguments

    if fn_name == "detect_objects":
        target = fn_args.get("target", "objects")
        chosen_id = fn_args.get("image_id", "")

        img_bytes = uploaded_images.get(chosen_id)
        if img_bytes is None:
            for entry in reversed(session_images.get(session_id, [])):
                if entry["id"] in uploaded_images:
                    img_bytes = uploaded_images[entry["id"]]
                    break

        if img_bytes is None:
            conversations[session_id].append({
                "role": "tool",
                "content": "No image available for detection.",
            })
            await ws.send_text(json.dumps({
                "type": "error",
                "text": "No image found to run detection on.",
            }))
            return

        await ws.send_text(json.dumps({
            "type": "status",
            "text": f"Running detection for '{target}'...",
        }))

        det = await asyncio.to_thread(tools.run_detection, img_bytes, target, model)

        conversations[session_id].append({
            "role": "tool",
            "content": json.dumps({
                "bboxes": det["bboxes"],
                "count": det["count"],
                "target": det["target"],
                "detection_time": f"{det['detection_time_s']}s",
            }),
        })

        await ws.send_text(json.dumps({
            "type": "detection",
            "image_url": det["image_url"],
            "target": det["target"],
            "count": det["count"],
            "bboxes": det["bboxes"],
            "detection_time_s": det["detection_time_s"],
        }))

    elif fn_name == "web_search":
        query = fn_args.get("query", "")

        await ws.send_text(json.dumps({
            "type": "status",
            "text": f"Searching the web for '{query}'...",
        }))

        res = await asyncio.to_thread(tools.run_web_search, query, 5)

        conversations[session_id].append({
            "role": "tool",
            "content": json.dumps({"query": res["query"], "results": res["results"]}),
        })

        await ws.send_text(json.dumps({
            "type": "web_search",
            "query": res["query"],
            "results": res["results"],
            "search_time_s": res["search_time_s"],
        }))

    else:
        # Unknown tool — return an error to the model so it can adjust
        conversations[session_id].append({
            "role": "tool",
            "content": f"Unknown tool: {fn_name}",
        })


async def _generate_response(ws: WebSocket, session_id: str, use_tools: bool = True, has_audio: bool = False):
    """Generate a response, optionally checking for tool calls first."""
    model = session_models.get(session_id, DEFAULT_MODEL)
    print(f"[session {session_id[:8]}] generating with model={model}, tools={use_tools}, audio={has_audio}")
    messages = conversations[session_id]

    # Build system message with available images so the model can reference them by ID
    imgs = session_images.get(session_id, [])
    system_parts = []
    if imgs and use_tools:
        img_list = ", ".join(f"image #{e['index']} (id: {e['id']})" for e in imgs)
        system_parts.append(
            f"The user has uploaded the following images in this conversation: {img_list}. "
            "When calling detect_objects, use the image_id of the relevant image."
        )
    system_msg = {"role": "system", "content": " ".join(system_parts)} if system_parts else None
    llm_messages = ([system_msg] + messages) if system_msg else messages

    # --- No-tools path: straight streaming (used for audio, or when tools aren't relevant) ---
    if not use_tools:
        # Audio requires num_ctx=8000 for reliable processing
        opts = {"num_ctx": 8000} if has_audio else None
        await _stream_response(ws, session_id, model, llm_messages, options=opts)
        return

    # --- Tools path: loop until the model stops calling tools ---
    MAX_TOOL_STEPS = 5
    start = time.perf_counter()
    final_response = None

    for step in range(MAX_TOOL_STEPS):
        # Rebuild llm_messages each iteration so it includes new tool results
        llm_messages = ([system_msg] + conversations[session_id]) if system_msg else conversations[session_id]

        response = await asyncio.to_thread(
            ollama_chat,
            model=model,
            messages=llm_messages,
            tools=TOOLS,
        )

        tool_calls = response.message.tool_calls

        if not tool_calls:
            # Model is done calling tools — this is the final response
            final_response = response
            break

        print(f"[session {session_id[:8]}] tool step {step + 1}/{MAX_TOOL_STEPS}: {[tc.function.name for tc in tool_calls]}")

        # Record the assistant's tool-call message
        conversations[session_id].append({
            "role": "assistant",
            "content": response.message.content or "",
            "tool_calls": [
                {"function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in tool_calls
            ],
        })

        # Execute each tool call and append its result
        for tc in tool_calls:
            await _execute_tool_call(ws, session_id, model, tc)
    else:
        # Hit max steps without the model finishing — force a final streaming response
        print(f"[session {session_id[:8]}] hit MAX_TOOL_STEPS, forcing final answer")
        await _stream_response(ws, session_id, model, conversations[session_id])
        return

    # If we got here, final_response is set (model returned no tool calls)
    if final_response is None:
        # Shouldn't happen, but stream as fallback
        await _stream_response(ws, session_id, model, conversations[session_id])
        return

    # If any tool calls happened, we need to stream the final answer (the non-streaming
    # final_response already has the full text — send it as one token for consistency)
    full_text = final_response.message.content or ""
    end = time.perf_counter()

    if full_text:
        await ws.send_text(json.dumps({"type": "token", "text": full_text}))

    conversations[session_id].append({"role": "assistant", "content": full_text})

    eval_count = getattr(final_response, "eval_count", None)
    prompt_eval_count = getattr(final_response, "prompt_eval_count", None)
    eval_duration_ns = getattr(final_response, "eval_duration", None)
    prompt_eval_duration_ns = getattr(final_response, "prompt_eval_duration", None)

    tok_per_sec = None
    if eval_count and eval_duration_ns and eval_duration_ns > 0:
        tok_per_sec = round(eval_count / (eval_duration_ns / 1e9), 1)
    ttft_s = None
    if prompt_eval_duration_ns:
        ttft_s = round(prompt_eval_duration_ns / 1e9, 2)

    stats = {
        "type": "stats",
        "model": model,
        "total_time_s": round(end - start, 2),
        "ttft_s": ttft_s or round(end - start, 2),
        "eval_tokens": eval_count,
        "prompt_tokens": prompt_eval_count,
        "tokens_per_sec": tok_per_sec,
    }
    await ws.send_text(json.dumps(stats))
    await ws.send_text(json.dumps({"type": "done"}))


# Mount static files last
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import os
    import socket as _socket
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    # Bind EXACTLY the requested interfaces (default just loopback). Binding one
    # socket per host — never 0.0.0.0 — means tailnet mode adds the Tailscale IP
    # WITHOUT ever exposing the wider local network. `run.sh --tailnet` sets
    # BIND_HOSTS="127.0.0.1,<100.x tailscale ip>".
    hosts = [h.strip() for h in os.environ.get("BIND_HOSTS", "127.0.0.1").split(",") if h.strip()]

    # Warm the voice models in the background so the first spoken turn is fast.
    print(f"[boot] brain={DEFAULT_MODEL}  voice={voice_engine.tts_label()}  warming models…")
    threading.Thread(target=voice_engine.warm, daemon=True).start()

    socks = []
    for h in hosts:
        fam = _socket.AF_INET6 if ":" in h else _socket.AF_INET
        s = _socket.socket(fam, _socket.SOCK_STREAM)
        s.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
        s.bind((h, port))
        s.listen()
        s.set_inheritable(True)
        socks.append(s)
        print(f"[boot] serving on  ->  http://{h}:{port}")
    uvicorn.Server(uvicorn.Config(app, log_level="warning")).run(sockets=socks)
