"""
Gemma 4 Chat Server
- WebSocket streaming chat with conversation history
- Image upload support (multimodal)
- Tool calling: detect_objects (bounding box detection)
- Throughput stats per response
"""

import asyncio
import base64
import io
import json
import subprocess
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from ollama import chat as ollama_chat
from PIL import Image, ImageDraw, ImageFont

from web_search import web_search

app = FastAPI()

# --- State ---
conversations: dict[str, list[dict]] = {}  # session_id -> messages
uploaded_images: dict[str, bytes] = {}  # image_id -> raw bytes
session_images: dict[str, list[dict]] = {}  # session_id -> [{id, filename, index}]

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
DETECTION_DIR = Path("detection")
DETECTION_DIR.mkdir(exist_ok=True)

MODEL_INFO = {
    "gemma4:e4b": {"audio": True},
    "gemma4:26b": {"audio": False},
}
AVAILABLE_MODELS = list(MODEL_INFO.keys())
DEFAULT_MODEL = "gemma4:e4b"
session_models: dict[str, str] = {}  # session_id -> model name

# --- Detection (from gemma_detect.py) ---

BBOX_SCHEMA = {
    "type": "object",
    "properties": {
        "bboxes": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 4,
                "maxItems": 4,
            },
        }
    },
    "required": ["bboxes"],
}


def run_detection(image_bytes: bytes, target: str, model: str = DEFAULT_MODEL) -> tuple[list, bytes, float]:
    """Run bbox detection on image bytes. Returns (bboxes, annotated_png_bytes, elapsed)."""
    # Save temp image for ollama
    tmp_path = UPLOAD_DIR / f"_detect_{uuid.uuid4().hex}.png"
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img.save(tmp_path, "PNG")

    prompt = (
        f"Detect all {target} in this image. "
        "Return bounding boxes as [y_min, x_min, y_max, x_max] "
        "with coordinates normalized to 0-1000."
    )

    start = time.perf_counter()
    response = ollama_chat(
        model=model,
        messages=[{"role": "user", "content": prompt, "images": [str(tmp_path)]}],
        format=BBOX_SCHEMA,
    )
    elapsed = time.perf_counter() - start

    result = json.loads(response.message.content)
    bboxes = result.get("bboxes", [])

    # Draw bboxes
    w, h = img.size
    draw = ImageDraw.Draw(img)
    for i, (y_min, x_min, y_max, x_max) in enumerate(bboxes):
        left = x_min / 1000 * w
        top_ = y_min / 1000 * h
        right = x_max / 1000 * w
        bottom = y_max / 1000 * h
        draw.rectangle([left, top_, right, bottom], outline="red", width=3)
        draw.text((left + 4, top_ - 14), f"#{i + 1}", fill="red")

    buf = io.BytesIO()
    img.save(buf, "PNG")
    annotated_bytes = buf.getvalue()

    # Cleanup temp
    tmp_path.unlink(missing_ok=True)

    return bboxes, annotated_bytes, elapsed


# --- Tool definition for the LLM ---

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "detect_objects",
            "description": (
                "Detect objects in an uploaded image and return bounding boxes. "
                "Use this when the user asks to find, detect, or locate objects in an image. "
                "You MUST specify the image_id of the image to run detection on."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "image_id": {
                        "type": "string",
                        "description": "The ID of the uploaded image to run detection on",
                    },
                    "target": {
                        "type": "string",
                        "description": "The object type to detect (e.g. 'people', 'cars', 'plates')",
                    },
                },
                "required": ["image_id", "target"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web via DuckDuckGo for current information. "
                "Use this for questions about recent events, current facts, "
                "or anything that requires up-to-date knowledge beyond your training data."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query",
                    },
                },
                "required": ["query"],
            },
        },
    },
]


# --- Routes ---

@app.get("/models")
async def list_models():
    return {"models": MODEL_INFO, "default": DEFAULT_MODEL}


@app.get("/")
async def index():
    return FileResponse("static/index.html")


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

            # Handle model switch
            if msg.get("type") == "set_model":
                new_model = msg.get("model", DEFAULT_MODEL)
                if new_model in AVAILABLE_MODELS:
                    session_models[session_id] = new_model
                print(f"[session {session_id[:8]}] model switched to {session_models[session_id]}")
                await ws.send_text(json.dumps({"type": "model_set", "model": session_models[session_id]}))
                continue

            text = msg.get("text", "")
            image_ids = msg.get("image_ids", [])
            audio_ids = msg.get("audio_ids", [])

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

            # Audio messages bypass tool-calling (tools mode breaks audio understanding)
            await _generate_response(ws, session_id, use_tools=not has_audio, has_audio=has_audio)

    except WebSocketDisconnect:
        pass


async def _stream_response(ws: WebSocket, session_id: str, model: str,
                           messages: list[dict], options: dict | None = None):
    """Pure streaming call — no tools. Used for audio and the post-tool follow-up."""
    import queue
    import threading

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

        bboxes, annotated_bytes, detect_elapsed = await asyncio.to_thread(
            run_detection, img_bytes, target, model
        )

        det_id = uuid.uuid4().hex
        det_path = DETECTION_DIR / f"{det_id}.png"
        det_path.write_bytes(annotated_bytes)

        conversations[session_id].append({
            "role": "tool",
            "content": json.dumps({
                "bboxes": bboxes,
                "count": len(bboxes),
                "target": target,
                "detection_time": f"{detect_elapsed:.2f}s",
            }),
        })

        await ws.send_text(json.dumps({
            "type": "detection",
            "image_url": f"/images/{det_id}",
            "target": target,
            "count": len(bboxes),
            "bboxes": bboxes,
            "detection_time_s": round(detect_elapsed, 2),
        }))

    elif fn_name == "web_search":
        query = fn_args.get("query", "")

        await ws.send_text(json.dumps({
            "type": "status",
            "text": f"Searching the web for '{query}'...",
        }))

        try:
            search_start = time.perf_counter()
            results = await asyncio.to_thread(web_search, query, 5)
            search_elapsed = time.perf_counter() - search_start
        except Exception as e:
            results = []
            search_elapsed = 0
            print(f"[web_search] error: {e}")

        conversations[session_id].append({
            "role": "tool",
            "content": json.dumps({"query": query, "results": results}),
        })

        await ws.send_text(json.dumps({
            "type": "web_search",
            "query": query,
            "results": results,
            "search_time_s": round(search_elapsed, 2),
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
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
