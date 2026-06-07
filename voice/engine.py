"""
Voice engine — speech in, speech out — Orpheus only, lifted from the
`real_time_voice` Orpheus build so the realtime voice experience is identical:

    browser mic (webm/mp4)  ->  ffmpeg -> 16k wav
                            ->  Whisper (MLX / Metal)        [speech -> text]
                            ->  Ollama chat model            [text -> reply]
                            ->  Orpheus TTS (emotion tags)   [reply -> speech]
                            ->  PCM streamed back to the browser

Everything runs on-device. This module owns the voice pipeline + its OWN short
spoken conversation history (kept separate from the rich text-chat history so the
voice turn stays snappy and concise — the way it has always worked).

Orpheus runs through Ollama (`orpheus-tts`) + a SNAC decoder. macOS `say` is kept
only as an emergency fallback if Orpheus errors.
"""

import io
import os
import re
import subprocess
import tempfile
import threading
import wave

import requests

# ----------------------------------------------------------------------------
# Config (override with env vars)
# ----------------------------------------------------------------------------
OLLAMA_URL    = os.environ.get("OLLAMA_URL", "http://localhost:11434")
WHISPER_REPO  = os.environ.get("WHISPER_REPO", "mlx-community/whisper-large-v3-turbo")
ORPHEUS_VOICE = os.environ.get("ORPHEUS_VOICE", "tara")          # Orpheus voice
# Default voice chat brain — gemma4:12b is the smart brain the realtime build ships.
VOICE_LLM_MODEL = os.environ.get("VOICE_LLM_MODEL", os.environ.get("LLM_MODEL", "gemma4:12b"))

# ----------------------------------------------------------------------------
# Emotion tags + text cleaning (Orpheus renders <laugh>/<sigh>/… inline; we ask
# the model to place them, then strip them from the on-screen text).
# ----------------------------------------------------------------------------
_TAG_RE = re.compile(r"\s*<(?:laugh|chuckle|giggle|sigh|gasp|yawn|groan|sniffle|cough)>\s*")

def strip_emotion_tags(text: str) -> str:
    return _TAG_RE.sub(" ", text).strip()


_EMOS = ["laugh", "chuckle", "giggle", "sigh", "gasp", "yawn", "groan", "sniffle", "cough"]
_EMO_FUZZ = re.compile(r"[<(\[*]\s*(" + "|".join(_EMOS) + r")[a-z]*\s*[,.!?]*\s*[>)\]*]?", re.I)
_STRAY_ANGLE = re.compile(r"<[^>\n]{0,24}>?")


def _normalize_emotion_tags(text: str) -> str:
    """Less-obedient models write sloppy tags: '<chuckle,' or '*chuckle*' or
    '(chuckles)'. Repair them to canonical <chuckle>, then drop any other stray
    <...> fragment so the voice never reads a broken tag aloud."""
    text = _EMO_FUZZ.sub(lambda m: " <" + m.group(1).lower() + "> ", text)
    text = _STRAY_ANGLE.sub(
        lambda m: m.group(0) if re.fullmatch(r"<(?:" + "|".join(_EMOS) + r")>", m.group(0)) else " ",
        text)
    return text


def clean_for_tts(text: str) -> str:
    """Normalize emotion tags + strip markdown / roleplay stage-directions so the
    voice doesn't read symbols, *actions*, or broken tags aloud. Keeps valid
    <emotion> tags intact."""
    text = _normalize_emotion_tags(text)
    text = re.sub(r"\*\*([^*\n]+)\*\*", r"\1", text)   # **bold** -> keep the word
    text = re.sub(r"\*([^*\n]+)\*", " ", text)         # *grins*   -> drop the action
    text = re.sub(r"_([^_\n]+)_", " ", text)           # _whispers_-> drop
    text = re.sub(r"[`#~|*]+", "", text)               # stray markdown (NOT < or >)
    return re.sub(r"[ \t]{2,}", " ", text).strip()


# The Orpheus system prompt — short spoken replies + the inline emotion tags
# Orpheus renders. This is the realtime Orpheus build's prompt verbatim.
SYSTEM_PROMPT = (
    "You are a friendly, concise voice assistant having a spoken conversation. "
    "Your replies are read aloud, so: keep them short (1-3 sentences), natural and "
    "conversational, and never use markdown, bullet points, code blocks, or emoji. "
    "Spell out things that should be spoken. If asked for a long answer, give the "
    "spoken-friendly short version and offer to go deeper."
    " Your expressive voice is slow to synthesize, so reply in ONE very short "
    "sentence (about 6 to 12 words) unless explicitly asked for more. To sound "
    "human, you may add occasional emotion cues "
    "using ONLY these exact tags: <laugh>, <chuckle>, <giggle>, <sigh>, <gasp>, "
    "<yawn>, <groan>, <sniffle>, <cough>. Use them sparingly, only where a real "
    "person naturally would, e.g. \"That's hilarious <laugh> I love it.\""
)

# ----------------------------------------------------------------------------
# Conversation state — the voice turn keeps its OWN short history so it stays
# snappy/concise. (A single local user -> one in-memory history is plenty.)
# ----------------------------------------------------------------------------
_history = [{"role": "system", "content": SYSTEM_PROMPT}]
_lock = threading.Lock()

_state = {
    "model": VOICE_LLM_MODEL,
    "voice": ORPHEUS_VOICE,
}


def current_voice() -> str:
    return _state["voice"]


# ----------------------------------------------------------------------------
# Speech-to-text  (MLX Whisper, lazily loaded + weight-cached by the library)
# ----------------------------------------------------------------------------
def transcribe(wav_path: str) -> str:
    import mlx_whisper
    result = mlx_whisper.transcribe(
        wav_path,
        path_or_hf_repo=WHISPER_REPO,
        fp16=True,
    )
    return (result.get("text") or "").strip()


# ----------------------------------------------------------------------------
# Text-to-speech  (Orpheus; macOS `say` only as an emergency fallback)
# ----------------------------------------------------------------------------
def synthesize(text: str) -> bytes:
    """Return WAV bytes for `text` via Orpheus (keeps emotion tags)."""
    text = text.strip()
    if not text:
        text = "Sorry, I didn't catch that."
    try:
        from . import orpheus_tts
        return orpheus_tts.synthesize(text, _state["voice"])
    except Exception as e:  # noqa: BLE001
        print(f"[tts] Orpheus failed ({e}); using `say`")
        return _say_wav(strip_emotion_tags(text))


def _say_wav(text: str) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        out = f.name
    subprocess.run(["say", "-o", out, "--data-format=LEI16@22050", text], check=True)
    with open(out, "rb") as fh:
        data = fh.read()
    os.unlink(out)
    return data


def _wav_to_pcm(wav_bytes: bytes):
    """WAV bytes -> (int16 PCM bytes, sample_rate). Assumes 16-bit mono."""
    with wave.open(io.BytesIO(wav_bytes)) as w:
        return w.readframes(w.getnframes()), w.getframerate()


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def sentences(text: str):
    """Split a reply into TTS-sized chunks, keeping emotion tags with their text."""
    text = text.strip()
    if not text:
        return []
    parts = [s for s in _SENT_SPLIT.split(text) if s.strip()]
    merged = []
    for p in parts:
        if merged and not strip_emotion_tags(p):
            merged[-1] = merged[-1] + " " + p
        else:
            merged.append(p)
    return merged


def tts_stream(text: str, voice: str = None):
    """Yield (pcm_int16_bytes, sample_rate) for `text`, streaming Orpheus windows
    (~85 ms) as the tokens arrive so playback starts almost immediately."""
    produced = False
    try:
        from . import orpheus_tts
        for chunk in orpheus_tts.synthesize_stream(text, voice or _state["voice"]):
            produced = True
            yield chunk, orpheus_tts.SAMPLE_RATE
    except Exception as e:  # noqa: BLE001
        print(f"[tts] Orpheus stream failed ({e}); using `say`")
        # If Orpheus already emitted 24k audio, do NOT also append `say` PCM.
    if not produced:
        yield _wav_to_pcm(_say_wav(strip_emotion_tags(text)))


# ----------------------------------------------------------------------------
# LLM  (Ollama chat API) — the voice brain. Short, snappy, think:false.
# ----------------------------------------------------------------------------
def chat(user_text: str) -> str:
    # Transactional: only commit the user turn alongside the assistant reply
    # AFTER success, so a failed turn can't leave a dangling user message.
    with _lock:
        messages = list(_history) + [{"role": "user", "content": user_text}]

    r = requests.post(
        f"{OLLAMA_URL}/api/chat",
        # think=false: gemma4:12b is a reasoning model; its hidden thinking pass
        # adds ~6s/turn that we don't want in a real-time voice loop.
        json={"model": _state["model"], "messages": messages, "stream": False,
              "think": False, "options": {"num_predict": 64}},
        timeout=120,
    )
    r.raise_for_status()
    reply = r.json()["message"]["content"].strip()

    with _lock:
        _history.append({"role": "user", "content": user_text})
        _history.append({"role": "assistant", "content": reply})  # keep emotion tags
    return reply


def reset_history():
    with _lock:
        del _history[1:]  # keep the system prompt


def get_system_prompt() -> str:
    with _lock:
        if _history and _history[0]["role"] == "system":
            return _history[0]["content"]
    return ""


def set_system_prompt(prompt: str):
    with _lock:
        if _history and _history[0]["role"] == "system":
            _history[0]["content"] = prompt
        else:
            _history.insert(0, {"role": "system", "content": prompt})


# ----------------------------------------------------------------------------
# Audio decode helper: any browser blob -> 16k mono wav via ffmpeg
# ----------------------------------------------------------------------------
def to_wav16k(src_bytes: bytes, suffix: str) -> str:
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(src_bytes)
        src = f.name
    dst = src + ".16k.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-i", src, "-ar", "16000", "-ac", "1", dst],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    os.unlink(src)
    return dst


# ----------------------------------------------------------------------------
# Voice config helpers (for the UI settings panel)
# ----------------------------------------------------------------------------
def available_voices():
    try:
        from . import orpheus_tts
        return list(orpheus_tts.AVAILABLE_VOICES)
    except Exception:  # noqa: BLE001
        return ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"]


def tts_label() -> str:
    return f"orpheus:{_state['voice']}"


def list_chat_models():
    """Every Ollama chat model usable as the brain (exclude the TTS model)."""
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        r.raise_for_status()
        names = [m["name"] for m in r.json().get("models", [])]
        usable = sorted(
            n for n in names
            if not n.lower().startswith(("orpheus", "snac")) and "tts" not in n.lower()
        )
        return usable or [_state["model"]]
    except Exception:  # noqa: BLE001
        return [_state["model"]]


def ollama_ok() -> bool:
    try:
        requests.get(f"{OLLAMA_URL}/api/tags", timeout=3).raise_for_status()
        return True
    except Exception:  # noqa: BLE001
        return False


def set_config(model=None, voice=None, system_prompt=None):
    """Live-update the voice brain / voice / system prompt from the UI."""
    if model:
        _state["model"] = model
    if voice:
        _state["voice"] = voice
    if system_prompt is not None:
        set_system_prompt(system_prompt)
    return dict(_state)


def warm():
    """Pre-load Whisper + Orpheus + the LLM so the first spoken turn is fast."""
    try:
        from . import orpheus_tts
        orpheus_tts.synthesize("Hello there.", _state["voice"])
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        subprocess.run(["say", "-o", tmp, "--data-format=LEI16@22050", "ready"], check=True)
        transcribe(tmp)
        os.unlink(tmp)
        requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={"model": _state["model"], "messages": [{"role": "user", "content": "hi"}],
                  "stream": False},
            timeout=120,
        )
        print("[warm] voice models ready — first turn will be fast")
    except Exception as e:  # noqa: BLE001
        print(f"[warm] warm-up skipped: {e}")
