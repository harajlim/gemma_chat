"""
Orpheus TTS, fully local via Ollama + SNAC.

Orpheus is a Llama-3B speech model that emits audio as special tokens
(<custom_token_N>). We run the GGUF inside Ollama (raw completion, no chat
template), stream those tokens, turn them into SNAC codes, and decode to 24 kHz
audio on the Mac GPU (mps). It understands inline emotion tags like <laugh>,
<sigh>, <gasp> — which is the whole point of this trial.

Decode logic adapted from isaiahbjork/orpheus-tts-local (MIT).
"""
import io
import os
import wave

import numpy as np
import requests

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

OLLAMA_URL    = os.environ.get("OLLAMA_URL", "http://localhost:11434")
ORPHEUS_MODEL = os.environ.get("ORPHEUS_MODEL", "orpheus-tts")
SAMPLE_RATE   = 24000

AVAILABLE_VOICES = ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"]
EMOTION_TAGS = ["<laugh>", "<chuckle>", "<giggle>", "<sigh>", "<gasp>",
                "<yawn>", "<groan>", "<sniffle>", "<cough>"]
CUSTOM_TOKEN_PREFIX = "<custom_token_"

# ---- SNAC decoder (loaded once) -----------------------------------------
_snac = None
_device = None


def _snac_model():
    # Decode on the GPU (MPS) — ~11ms/decode vs ~169ms on CPU — which is what
    # makes per-window streaming run faster than real-time. Falls back to CPU.
    global _snac, _device
    if _snac is None:
        from snac import SNAC
        pref = os.environ.get("ORPHEUS_SNAC_DEVICE", "")
        if not pref:
            try:
                import torch
                pref = "mps" if torch.backends.mps.is_available() else "cpu"
            except Exception:  # noqa: BLE001
                pref = "cpu"
        _device = pref
        _snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(_device)
        print(f"[orpheus] SNAC decoder ready on {_device}")
    return _snac


# ---- token -> SNAC code id ----------------------------------------------
def _token_to_id(token_text: str, index: int):
    token_text = token_text.strip()
    pos = token_text.rfind(CUSTOM_TOKEN_PREFIX)
    if pos == -1:
        return None
    last = token_text[pos:]
    if last.startswith(CUSTOM_TOKEN_PREFIX) and last.endswith(">"):
        try:
            return int(last[14:-1]) - 10 - ((index % 7) * 4096)
        except ValueError:
            return None
    return None


def _decode_codes(ids):
    """Decode the FULL SNAC code sequence at once -> int16 PCM bytes.

    Decoding the whole utterance (rather than sliding 28-id windows) avoids the
    head/tail clipping that window-slicing introduces on every sentence. Codes
    are clamped to the valid [0, 4095] range so one stray token can't drop the
    whole sentence (or crash the embedding lookup).
    """
    import torch
    model = _snac_model()  # also sets _device — must run before building tensors
    n = len(ids) // 7
    if n == 0:
        return b""
    frame = ids[: n * 7]
    c0, c1, c2 = [], [], []
    for j in range(n):
        i = 7 * j
        c0.append(frame[i])
        c1.extend([frame[i + 1], frame[i + 4]])
        c2.extend([frame[i + 2], frame[i + 3], frame[i + 5], frame[i + 6]])

    def _codes(vals):
        arr = np.clip(np.asarray(vals, dtype=np.int64), 0, 4095)
        return torch.tensor(arr, device=_device, dtype=torch.int32).unsqueeze(0)

    codes = [_codes(c0), _codes(c1), _codes(c2)]
    with torch.inference_mode():
        audio = model.decode(codes)
    audio = audio.squeeze().detach().cpu().numpy()
    return (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()


# ---- stream tokens from Ollama (raw, no chat template) ------------------
def _stream_tokens(prompt: str, max_tokens: int = 1200):
    payload = {
        "model": ORPHEUS_MODEL,
        "prompt": prompt,
        "raw": True,
        "stream": True,
        "options": {
            "temperature": 0.6,
            "top_p": 0.9,
            "repeat_penalty": 1.1,
            "num_predict": max_tokens,
            "num_ctx": 2048,  # Orpheus emits <=1200 audio tokens; no need for 32k KV cache
            "stop": ["<|eot_id|>"],
        },
    }
    import json
    with requests.post(f"{OLLAMA_URL}/api/generate", json=payload, stream=True, timeout=300) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            obj = json.loads(line.decode("utf-8"))
            if obj.get("response"):
                yield obj["response"]
            if obj.get("done"):
                break


# ---- public API ----------------------------------------------------------
def _collect_ids(text: str, voice: str):
    """Stream tokens from Orpheus and collect SNAC code ids for the utterance.

    `tid >= 0` (not `> 0`): code id 0 is legitimate, and dropping it would
    desync the 7-codebook frame alignment used by the (index % 7) de-offset.
    """
    if voice not in AVAILABLE_VOICES:
        voice = "tara"
    prompt = f"<|audio|>{voice}: {text}<|eot_id|>"
    ids = []
    count = 0
    for tok in _stream_tokens(prompt):
        tid = _token_to_id(tok, count)
        if tid is not None and tid >= 0:
            ids.append(tid)
            count += 1
    return ids


def _window_pcm(ids28, head=False):
    """Decode one sliding 28-id (4-frame) window -> int16 PCM bytes.
    Emits the [2048:4096] slice (contiguous body); on the first window emit
    [0:4096] so the utterance's onset isn't clipped."""
    import torch
    model = _snac_model()
    n = len(ids28) // 7
    if n == 0:
        return b""
    frame = ids28[: n * 7]
    c0, c1, c2 = [], [], []
    for j in range(n):
        i = 7 * j
        c0.append(frame[i])
        c1.extend([frame[i + 1], frame[i + 4]])
        c2.extend([frame[i + 2], frame[i + 3], frame[i + 5], frame[i + 6]])

    def _c(vals):
        arr = np.clip(np.asarray(vals, dtype=np.int64), 0, 4095)
        return torch.tensor(arr, device=_device, dtype=torch.int32).unsqueeze(0)

    with torch.inference_mode():
        audio = model.decode([_c(c0), _c(c1), _c(c2)])
    audio = audio.squeeze().detach().cpu().numpy()
    audio = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    return audio[(0 if head else 2048):4096].tobytes()


def synthesize_stream(text: str, voice: str = "tara"):
    """Stream the utterance's audio in ~85 ms chunks AS tokens arrive (windowed
    SNAC decode), so playback can start almost immediately and run continuously
    — no per-sentence pauses. O(n): each window is a small fixed decode."""
    if voice not in AVAILABLE_VOICES:
        voice = "tara"
    prompt = f"<|audio|>{voice}: {text}<|eot_id|>"
    buffer = []
    count = 0
    first = True
    for tok in _stream_tokens(prompt):
        tid = _token_to_id(tok, count)
        if tid is not None and tid >= 0:
            buffer.append(tid)
            count += 1
            if count % 7 == 0 and count >= 28:
                pcm = _window_pcm(buffer[-28:], head=first)
                if pcm:
                    yield pcm
                    first = False


def synthesize(text: str, voice: str = "tara") -> bytes:
    """text (may contain emotion tags) -> full WAV bytes (24 kHz mono)."""
    pcm = _decode_codes(_collect_ids(text, voice))
    if not pcm:
        raise RuntimeError("Orpheus produced no audio (no <custom_token_> output)")

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(bytes(pcm))
    return buf.getvalue()


if __name__ == "__main__":
    # quick self-test
    import sys
    t = sys.argv[1] if len(sys.argv) > 1 else "Oh wow <laugh> that is wonderful to hear!"
    data = synthesize(t)
    out = "/tmp/lvc/orpheus_test.wav"
    os.makedirs("/tmp/lvc", exist_ok=True)
    with open(out, "wb") as f:
        f.write(data)
    print(f"wrote {len(data)} bytes -> {out}")
