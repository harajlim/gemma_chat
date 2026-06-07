# Gemma 4 — Chat + Voice

A single, fully-local app for Google's **Gemma 4** models (via [Ollama](https://ollama.com)) that
does two things in one cohesive surface:

- **Rich text chat** — streaming markdown, image upload & multimodal, **tool calling** (live web
  search + object detection with bounding boxes), throughput stats, a context meter.
- **Realtime voice** — speech-to-text (Whisper), the Gemma brain, and streamed **Orpheus** TTS
  (expressive, with inline emotion tags), played back gaplessly — the realtime_voice experience.

Two toggles give a 2×2 of input × output:

| | **Speak: off** | **Speak: on** |
|---|---|---|
| **Mic: off** | type → text reply (rich chat, tools) | type → **spoken** reply |
| **Mic: on** | talk → text reply | talk → **spoken** reply |

The two **Speak-on** paths use realtime's exact endpoints (`/api/converse_text`,
`/api/converse_stream`), so spoken replies are identical in quality to the realtime_voice app.

Everything runs on-device. The "brain" — **`gemma4:12b`** by default — is shared: pick a model once
and both the text chat and the voice loop use it.

This is the merge of two earlier apps (`testing_gemma` chat + `real_time_voice`). The voice pipeline
is preserved behaviour-for-behaviour; the chat keeps all its tools; the UI is new and unified.

---

## Quickstart

```bash
cd ~/Desktop/testing_gemma
./run.sh                 # → http://localhost:8000
```

Open **http://localhost:8000**, then either **type** (rich chat) or **tap the mic** (talk).
First launch warms the models (~20–30 s); after that, turns are fast.

### Reach it from your phone (Tailscale)

```bash
./run.sh --tailnet       # also serves on this machine's Tailscale IP (tailnet only)
```

It binds `127.0.0.1` **and** your `100.x` Tailscale IP — **never** `0.0.0.0`, so only this laptop and
your privileged tailnet devices can reach it (not the wider local network). It prints the URL.

- **Typing + hearing replies** works over plain http from your phone.
- **The mic** needs a secure context (browsers block `getUserMedia` off https/localhost). To *talk*
  from your phone, run `tailscale serve 8000` and open the `https://…` URL it gives you.

Knobs: `PORT=8010 ./run.sh`, `VENV_PYTHON=/path/to/python ./run.sh`,
`ORPHEUS_VOICE=leo ./run.sh`, `VOICE_LLM_MODEL=gemma4:e4b ./run.sh` (snappier voice brain).

---

## Prerequisites

1. **Ollama** running, with the models pulled:
   ```bash
   ollama list   # should include gemma4:12b (+ optionally e4b/e2b/26b) and orpheus-tts
   ```
2. **ffmpeg** (`brew install ffmpeg`) — converts browser mic audio to 16 kHz WAV.
3. **Python 3.12** with both dependency stacks (see below).

### The environment

The app needs **both** stacks: the chat stack (`ollama`, `Pillow`, `ddgs`) **and** the voice ML stack
(`mlx-whisper` for STT, and `snac`+`torch` for Orpheus TTS, plus `numpy`/`soundfile`).

On this machine that's the `real_time_voice` venv with the chat deps added — which is exactly what
`run.sh` points at by default (`VENV_PYTHON` overrides it). To build a fresh combined env instead:

```bash
python3.12 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
VENV_PYTHON=venv/bin/python ./run.sh
```

Orpheus runs the GGUF through Ollama (`orpheus-tts`) and decodes audio tokens with SNAC on the GPU —
no large model files to copy. Whisper weights come from the HuggingFace cache.

---

## Using it

- **Text chat** — type and press Enter. Markdown renders live with throughput stats.
- **Images** — click the image icon, drag-and-drop, or paste from the clipboard.
- **Object detection** — upload an image and ask *"find all the people in this image"*; the model
  calls `detect_objects` and renders bounding boxes inline.
- **Web search** — ask about current events; the model calls `web_search` (DuckDuckGo) and answers
  from the results (shown in a collapsible card).
- **Voice** — flip **Mic** on to talk hands-free: it listens, detects when you stop, and replies.
  Flip **Speak** on to hear replies aloud (Orpheus). With Mic on it re-arms after each turn; flip it
  off to stop. The 2×2 of Mic × Speak is the table at the top.
- **Settings (gear)** — Orpheus voice, the voice "personality" (spoken-reply system prompt), and
  mic sensitivity.
- **Model picker** — the shared brain; `gemma4:12b` by default. Switching it points both chat and
  voice at the new model.
- **Voice playground** — `/tts` (also linked from settings): type text with emotion tags and hear it.

---

## Architecture

```
                       ┌─────────────────────────── Browser (static/) ───────────────────────────┐
                       │  index.html · app.js · style.css   —  one thread, two input paths        │
                       └───────────┬─────────────────────────────────────────────┬───────────────┘
                      WebSocket    │ /ws/chat                       REST NDJSON    │ /api/converse_stream
                       (text)      ▼                                  (voice)      ▼
                       ┌───────────────────────────  FastAPI server.py  ───────────────────────────┐
                       │  tool-calling chat loop          │       voice/routes.py → voice/engine.py │
                       │  (web_search, detect_objects)    │       Whisper STT · brain ·  Orpheus TTS  │
                       └───────────┬──────────────────────┴──────────────────┬──────────────────────┘
                                   ▼                                          ▼
                          ┌────────────────┐  ┌────────┐  ┌────────┐   ┌──────────┐  ┌──────────────┐
                          │ Ollama gemma4  │  │  DDGS  │  │ ffmpeg │   │ MLX Whisper │ Orpheus TTS  │
                          └────────────────┘  └────────┘  └────────┘   └──────────┘  └──────────────┘
```

| File | Purpose |
|------|---------|
| `server.py` | FastAPI app: WS tool-calling chat, image/detection, model picker; mounts the voice routes |
| `voice/engine.py` | STT + TTS + the spoken-conversation brain (lifted from the voice app — behaviour-identical) |
| `voice/routes.py` | Voice REST endpoints: `/api/converse_stream`, `/api/converse_text`, `/api/speak`, `/api/voice/*` |
| `voice/orpheus_tts.py` | Orpheus-over-Ollama + SNAC decoder (expressive TTS) |
| `web_search.py` | DuckDuckGo search tool |
| `static/` | Unified dark UI (chat + Mic/Speak toggles), the `/tts` playground |
| `run.sh` | Launcher (runs through the venv; `--tailnet` for phone access) |
| `tests/` | Unit + integration tests (LLM/TTS/STT mocked) |

---

## Tests

```bash
VENV_PYTHON=/path/to/python   # the combined env
"$VENV_PYTHON" -m pytest tests/ -q
```

Covers the voice engine's pure logic (text cleaning, emotion tags, sentence splitting, transactional
history) and the server (model picker incl. `gemma4:12b`, the WS chat path, a tool-calling turn, and
the voice NDJSON turn protocol) — all with Ollama/TTS/STT mocked, so it runs without models.

## Notes

- The voice brain uses `think:false` (gemma4:12b is a reasoning model — its hidden thinking pass adds
  ~6 s/turn) and short, spoken-friendly replies, so the realtime loop stays snappy.
- Detection boxes use Gemma's native 0–1000 normalized coordinates, drawn server-side with PIL.
- Nothing leaves the machine at inference time; `run.sh` sets `HF_HUB_OFFLINE=1`.
