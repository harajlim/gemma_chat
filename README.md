# Gemma 4 Chat

A browser-based chat interface for Google's Gemma 4 models, running locally via [Ollama](https://ollama.com). Supports text, image, and audio input with tool calling for web search and object detection.

## Features

- **Streaming chat** with token-by-token rendering and full markdown support
- **Multimodal input** — upload images, record audio (e4b only), or both
- **Tool calling** with multi-step reasoning (up to 5 rounds):
  - `web_search` — live DuckDuckGo search
  - `detect_objects` — draws bounding boxes on uploaded images
- **Model picker** — switch between `gemma4:e4b` (edge, with audio) and `gemma4:26b` (larger, vision only)
- **Live throughput stats** per response (TTFT, tokens/sec, total time)
- **Context meter** showing token usage against the 131K context window
- **Session-scoped conversation history** with image tracking

## Prerequisites

1. **Ollama** installed and running — [download here](https://ollama.com/download)
2. **Python 3.10+**
3. **ffmpeg** (only required if you want audio input) — `brew install ffmpeg` on macOS

Pull at least one Gemma 4 model:

```bash
ollama pull gemma4:e4b
# optional — larger model, no audio support:
ollama pull gemma4:26b
```

## Install

```bash
git clone git@github.com:harajlim/gemma_chat.git
cd gemma_chat

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run

Make sure Ollama is running in the background, then:

```bash
python server.py
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

## Usage

- **Text chat** — just type and press Enter
- **Images** — click the image icon, drag-and-drop onto the window, or paste from clipboard
- **Audio** — click the mic icon to record (only enabled when the selected model supports audio, i.e. `gemma4:e4b`)
- **Object detection** — upload an image and ask the model to find something, e.g. *"find all the people in this image"* — the model will call the `detect_objects` tool and render the image with bounding boxes inline
- **Web search** — ask about current events or recent info, e.g. *"what's the latest news on Gemma 4?"* — the model will call `web_search` and synthesize an answer from the results
- **Multi-step reasoning** — the model can chain tool calls, e.g. *"search for the top object detection benchmarks, then find cars in this image"*

## Architecture

```
┌─────────────────┐      WebSocket      ┌──────────────┐
│  Browser (JS)   │ ◄─────────────────► │  FastAPI     │
│  static/index   │                     │  server.py   │
└─────────────────┘                     └──────┬───────┘
                                               │
                                    ┌──────────┼──────────┐
                                    ▼          ▼          ▼
                              ┌─────────┐ ┌────────┐ ┌─────────┐
                              │ Ollama  │ │  DDGS  │ │ ffmpeg  │
                              │ Gemma 4 │ │ search │ │ convert │
                              └─────────┘ └────────┘ └─────────┘
```

- **`server.py`** — FastAPI server, WebSocket streaming, tool dispatch, conversation state
- **`web_search.py`** — DuckDuckGo search wrapper
- **`static/index.html`** — Single-file chat UI (HTML + CSS + vanilla JS with marked.js for markdown)

## Files

| File | Purpose |
|------|---------|
| `server.py` | FastAPI server with WebSocket chat, tool calling, image/audio handling |
| `web_search.py` | DuckDuckGo search function exposed as an LLM tool |
| `static/index.html` | Browser chat UI |
| `requirements.txt` | Python dependencies |

## Notes

- Audio input is only supported by the `e4b`/`e2b` edge models. The mic button is disabled when a vision-only model (like `26b`) is selected.
- Audio recordings are converted to 16 kHz mono WAV via ffmpeg for Ollama compatibility.
- The `num_ctx` parameter is set to 8000 when processing audio to satisfy Ollama's memory allocation requirements.
- Detection bounding boxes use normalized 0-1000 coordinates (Gemma's native output format) and are rendered onto the image server-side with PIL.
