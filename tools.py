"""
Shared tools for BOTH the text-chat (WS) path and the voice path:
  • web_search    — DuckDuckGo
  • detect_objects — Gemma bbox detection drawn onto the image with PIL

Keeping the schema + execution here means voice mode and chat mode call the exact
same tools, and the voice path can speak a short "what I'm doing" announcement
when one fires.
"""

import io
import json
import re
import time
import uuid
from pathlib import Path

from ollama import chat as ollama_chat
from PIL import Image, ImageDraw

from web_search import web_search as _ddg_search

UPLOAD_DIR = Path("uploads"); UPLOAD_DIR.mkdir(exist_ok=True)
DETECTION_DIR = Path("detection"); DETECTION_DIR.mkdir(exist_ok=True)

# Uploaded media ids are uuid4().hex — validate before any filesystem glob.
_ID_RE = re.compile(r"^[0-9a-f]{32}$")


def valid_id(s) -> bool:
    return isinstance(s, str) and bool(_ID_RE.match(s))


def resolve_image_bytes(image_id: str):
    """image_id -> raw bytes from uploads/, or None. Path-traversal safe."""
    if not valid_id(image_id):
        return None
    for p in UPLOAD_DIR.glob(f"{image_id}.*"):
        try:
            return p.read_bytes()
        except Exception:  # noqa: BLE001
            return None
    return None


BBOX_SCHEMA = {
    "type": "object",
    "properties": {
        "bboxes": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "integer"}, "minItems": 4, "maxItems": 4},
        }
    },
    "required": ["bboxes"],
}

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
                    "image_id": {"type": "string", "description": "The ID of the uploaded image to run detection on"},
                    "target": {"type": "string", "description": "The object type to detect (e.g. 'people', 'cars', 'plates')"},
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
                "properties": {"query": {"type": "string", "description": "The search query"}},
                "required": ["query"],
            },
        },
    },
]


def run_detection(image_bytes: bytes, target: str, model: str) -> dict:
    """Run bbox detection, write the annotated PNG, and return a UI-ready dict:
    {target, count, bboxes, detection_time_s, image_url}."""
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

    bboxes = json.loads(response.message.content).get("bboxes", [])
    w, h = img.size
    draw = ImageDraw.Draw(img)
    for i, (y_min, x_min, y_max, x_max) in enumerate(bboxes):
        draw.rectangle(
            [x_min / 1000 * w, y_min / 1000 * h, x_max / 1000 * w, y_max / 1000 * h],
            outline="red", width=3,
        )
        draw.text((x_min / 1000 * w + 4, y_min / 1000 * h - 14), f"#{i + 1}", fill="red")
    tmp_path.unlink(missing_ok=True)

    det_id = uuid.uuid4().hex
    buf = io.BytesIO(); img.save(buf, "PNG")
    (DETECTION_DIR / f"{det_id}.png").write_bytes(buf.getvalue())

    return {
        "target": target, "count": len(bboxes), "bboxes": bboxes,
        "detection_time_s": round(elapsed, 2), "image_url": f"/images/{det_id}",
    }


def run_web_search(query: str, n: int = 5) -> dict:
    """Return {query, results, search_time_s}."""
    start = time.perf_counter()
    try:
        results = _ddg_search(query, n)
    except Exception as e:  # noqa: BLE001
        print(f"[web_search] error: {e}")
        results = []
    return {"query": query, "results": results, "search_time_s": round(time.perf_counter() - start, 2)}


def announce(name: str, args: dict) -> str:
    """A short, spoken-friendly line the voice says while a tool runs."""
    if name == "web_search":
        q = (args or {}).get("query", "").strip()
        return f"Let me search the web for {q}." if q else "Let me search the web for that."
    if name == "detect_objects":
        t = (args or {}).get("target", "").strip()
        return f"Let me look for {t} in your image." if t else "Let me take a look at your image."
    return "One moment, let me look into that."
