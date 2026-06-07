"""Pure-logic unit tests for the voice engine — no models, no network.

These lock down the text-cleaning / sentence-splitting / config behaviour that
keeps the voice experience identical to the original real_time_voice app.
"""
import importlib

import pytest

engine = importlib.import_module("voice.engine")


# --- emotion tag stripping -------------------------------------------------
def test_strip_emotion_tags_removes_known_tags():
    assert engine.strip_emotion_tags("Hi <laugh> there") == "Hi there"
    assert engine.strip_emotion_tags("<sigh> ok") == "ok"
    assert engine.strip_emotion_tags("done <chuckle>") == "done"


def test_strip_emotion_tags_keeps_non_tags():
    # angle content that isn't a known emotion tag is left alone by strip_*
    assert engine.strip_emotion_tags("a < b and c > d") == "a < b and c > d"


# --- clean_for_tts: drop markdown/stage directions, keep emotion tags ------
def test_clean_for_tts_strips_markdown_keeps_emotion():
    out = engine.clean_for_tts("**Hello** *waves* world <laugh> and `code`")
    assert "Hello" in out
    assert "world" in out
    assert "code" in out          # backticks removed, word kept
    assert "*" not in out          # stage direction removed
    assert "<laugh>" in out        # valid emotion tag preserved


def test_clean_for_tts_repairs_sloppy_tags():
    # '(chuckles)' / '*chuckle*' should normalize to canonical <chuckle>
    assert "<chuckle>" in engine.clean_for_tts("That's funny (chuckles)")
    assert "<sigh>" in engine.clean_for_tts("well *sigh* fine")


def test_clean_for_tts_drops_stray_angle_fragments():
    out = engine.clean_for_tts("hello <unknownthing> world")
    assert "<unknownthing>" not in out
    assert "hello" in out and "world" in out


# --- sentence splitting ----------------------------------------------------
def test_sentences_basic_split():
    assert engine.sentences("One. Two! Three?") == ["One.", "Two!", "Three?"]


def test_sentences_merges_trailing_emotion_only_chunk():
    # a chunk that is only an emotion tag merges into the previous sentence
    out = engine.sentences("That is great. <laugh>")
    assert out == ["That is great. <laugh>"]


def test_sentences_empty():
    assert engine.sentences("   ") == []


# --- config / state --------------------------------------------------------
def test_set_config_switches_voice_and_model():
    engine.set_config(voice="leo", model="gemma4:e4b")
    assert engine.current_voice() == "leo"
    assert engine._state["model"] == "gemma4:e4b"
    # restore defaults for other tests
    engine.set_config(voice="tara", model="gemma4:12b")
    assert engine.current_voice() == "tara"


def test_set_config_system_prompt_roundtrip():
    engine.set_system_prompt("Be a pirate.")
    assert engine.get_system_prompt() == "Be a pirate."
    # reset back to default
    engine.set_system_prompt(engine.SYSTEM_PROMPT)


def test_reset_history_keeps_system_prompt():
    engine.set_system_prompt("keep me")
    engine._history.append({"role": "user", "content": "hi"})
    engine._history.append({"role": "assistant", "content": "yo"})
    engine.reset_history()
    assert len(engine._history) == 1
    assert engine._history[0]["role"] == "system"
    assert engine._history[0]["content"] == "keep me"
    engine.set_system_prompt(engine.SYSTEM_PROMPT)


def test_available_voices_orpheus():
    voices = engine.available_voices()
    assert "tara" in voices


# --- model listing (mock ollama) ------------------------------------------
def test_list_chat_models_excludes_tts(monkeypatch):
    class FakeResp:
        def raise_for_status(self): pass
        def json(self): return {"models": [
            {"name": "gemma4:12b"}, {"name": "gemma4:e4b"},
            {"name": "orpheus-tts:latest"}, {"name": "some-tts-model"},
        ]}

    monkeypatch.setattr(engine.requests, "get", lambda *a, **k: FakeResp())
    models = engine.list_chat_models()
    assert "gemma4:12b" in models
    assert "gemma4:e4b" in models
    assert not any("orpheus" in m for m in models)
    assert not any("tts" in m for m in models)


def test_chat_is_transactional(monkeypatch):
    """A failed ollama call must NOT leave a dangling user turn in history."""
    engine.reset_history()
    base_len = len(engine._history)

    def boom(*a, **k):
        raise RuntimeError("ollama down")

    monkeypatch.setattr(engine.requests, "post", boom)
    with pytest.raises(RuntimeError):
        engine.chat("hello?")
    assert len(engine._history) == base_len  # no partial state


def test_chat_commits_on_success(monkeypatch):
    engine.reset_history()
    base_len = len(engine._history)

    class FakeResp:
        def raise_for_status(self): pass
        def json(self): return {"message": {"content": "Hi there!"}}

    monkeypatch.setattr(engine.requests, "post", lambda *a, **k: FakeResp())
    reply = engine.chat("hello")
    assert reply == "Hi there!"
    assert len(engine._history) == base_len + 2
    assert engine._history[-2] == {"role": "user", "content": "hello"}
    assert engine._history[-1]["role"] == "assistant"
    engine.reset_history()
