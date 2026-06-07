"""Integration tests for the merged server — FastAPI TestClient with the LLM,
TTS and STT mocked, so they run fast and deterministically (no Ollama/models).

Covers: model picker (incl. gemma4:12b), the WS text-chat path, a tool-calling
turn (web_search), and the realtime-voice NDJSON turn protocol.
"""
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import server
from voice import engine


@pytest.fixture
def client():
    return TestClient(server.app)


# --- helpers to fake ollama responses --------------------------------------
def _resp(content="", tool_calls=None, **stats):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(message=msg, **stats)


def _toolcall(name, args):
    return SimpleNamespace(function=SimpleNamespace(name=name, arguments=args))


# --- model picker ----------------------------------------------------------
def test_models_endpoint_has_12b_and_default(client):
    data = client.get("/models").json()
    assert data["default"] == "gemma4:12b"
    assert "gemma4:12b" in data["models"]
    # all gemma models are vision-capable in this build
    assert data["models"]["gemma4:12b"]["vision"] is True


# --- WS plain text turn ----------------------------------------------------
def test_ws_text_turn(monkeypatch, client):
    # model returns no tool calls -> final answer in one shot
    monkeypatch.setattr(server, "ollama_chat",
                        lambda *a, **k: _resp(content="Hello from Gemma!",
                                              eval_count=3, prompt_eval_count=5,
                                              eval_duration=1_000_000_000,
                                              prompt_eval_duration=200_000_000))

    with client.websocket_connect("/ws/chat") as ws:
        ws.send_json({"session_id": "test-sess"})
        assert ws.receive_json()["type"] == "session"
        ws.send_json({"text": "hi"})

        seen = []
        while True:
            m = ws.receive_json()
            seen.append(m["type"])
            if m["type"] == "token":
                assert m["text"] == "Hello from Gemma!"
            if m["type"] == "done":
                break
    assert "token" in seen and "stats" in seen and "done" in seen


# --- WS tool-calling turn (web_search) -------------------------------------
def test_ws_tool_turn_web_search(monkeypatch, client):
    calls = {"n": 0}

    def fake_chat(*a, **k):
        # 1st call -> ask for web_search; 2nd call -> final grounded answer
        calls["n"] += 1
        if calls["n"] == 1:
            return _resp(content="", tool_calls=[_toolcall("web_search", {"query": "gemma 4"})])
        return _resp(content="Gemma 4 is great.", eval_count=4,
                     eval_duration=1_000_000_000, prompt_eval_duration=100_000_000)

    monkeypatch.setattr(server, "ollama_chat", fake_chat)
    monkeypatch.setattr(server, "web_search",
                        lambda q, n=5: [{"title": "T", "url": "http://x", "snippet": "s"}])

    with client.websocket_connect("/ws/chat") as ws:
        ws.send_json({"session_id": "tool-sess"})
        ws.receive_json()  # session
        ws.send_json({"text": "what's new with gemma 4?"})

        seen, search_payload = [], None
        while True:
            m = ws.receive_json()
            seen.append(m["type"])
            if m["type"] == "web_search":
                search_payload = m
            if m["type"] == "done":
                break

    assert "status" in seen          # "Searching the web…"
    assert "web_search" in seen
    assert search_payload["query"] == "gemma 4"
    assert search_payload["results"][0]["url"] == "http://x"
    assert "token" in seen           # final grounded answer streamed
    assert calls["n"] == 2


# --- WS model switch keeps voice brain in sync -----------------------------
def test_ws_set_model_syncs_voice_brain(monkeypatch, client):
    with client.websocket_connect("/ws/chat") as ws:
        ws.send_json({"session_id": "m-sess"})
        ws.receive_json()
        ws.send_json({"type": "set_model", "model": "gemma4:e4b"})
        m = ws.receive_json()
        assert m["type"] == "model_set" and m["model"] == "gemma4:e4b"
    assert engine._state["model"] == "gemma4:e4b"
    # restore
    engine.set_config(model="gemma4:12b")


# --- realtime voice: typed turn NDJSON protocol ----------------------------
def test_converse_text_ndjson(monkeypatch, client):
    monkeypatch.setattr(engine, "chat", lambda t: "Sure thing.")
    # one fake audio frame: 240 samples of int16 silence @ 24k
    monkeypatch.setattr(engine, "tts_stream",
                        lambda text, voice=None: iter([(b"\x00\x00" * 240, 24000)]))

    r = client.post("/api/converse_text", data={"text": "hello there"})
    assert r.status_code == 200
    events = [json.loads(l) for l in r.text.splitlines() if l.strip()]
    types = [e["type"] for e in events]

    assert types[0] == "transcript"
    assert events[0]["text"] == "hello there"
    assert "reply_text" in types
    reply = next(e for e in events if e["type"] == "reply_text")
    assert reply["text"] == "Sure thing."
    audio = [e for e in events if e["type"] == "audio"]
    assert audio and audio[0]["sr"] == 24000 and audio[0]["pcm"]
    assert types[-1] == "done"


def test_converse_text_empty_is_graceful(monkeypatch, client):
    r = client.post("/api/converse_text", data={"text": "   "})
    events = [json.loads(l) for l in r.text.splitlines() if l.strip()]
    types = [e["type"] for e in events]
    # empty transcript -> immediate done, no LLM/TTS
    assert types == ["transcript", "done"]


# --- robustness / security fixes ------------------------------------------
def test_id_regex_filters_traversal():
    assert server._ID_RE.match("0123456789abcdef0123456789abcdef")
    assert not server._ID_RE.match("../secret")
    assert not server._ID_RE.match("a" * 31)      # too short
    assert not server._ID_RE.match("A" * 32)      # uppercase (uuid hex is lower)
    assert not server._ID_RE.match("x" * 32)      # non-hex


def test_ws_filters_invalid_image_ids(monkeypatch, client):
    """Client-sent ids that aren't uuid4 hex must never reach a filesystem glob."""
    monkeypatch.setattr(server, "ollama_chat", lambda *a, **k: _resp(content="ok"))
    with client.websocket_connect("/ws/chat") as ws:
        ws.send_json({"session_id": "trav-sess"}); ws.receive_json()
        ws.send_json({"text": "look", "image_ids": ["../secret", "not-an-id"]})
        while ws.receive_json()["type"] != "done":
            pass
    # the bad ids were filtered out -> nothing tracked for the session
    assert server.session_images.get("trav-sess") == []


def test_ws_generation_error_sends_terminal_frame(monkeypatch, client):
    """If generation throws, the client MUST still get error + done (never hang) —
    this is what keeps the voice orb from getting wedged."""
    def boom(*a, **k):
        raise RuntimeError("ollama exploded")
    monkeypatch.setattr(server, "ollama_chat", boom)
    with client.websocket_connect("/ws/chat") as ws:
        ws.send_json({"session_id": "err-sess"}); ws.receive_json()
        ws.send_json({"text": "hi"})
        seen = []
        while True:
            m = ws.receive_json(); seen.append(m["type"])
            if m["type"] == "done":
                break
        assert "error" in seen
        assert seen[-1] == "done"


# --- voice config endpoints ------------------------------------------------
def test_voice_config_roundtrip(client):
    r = client.post("/api/voice/config", json={"voice": "leo"})
    assert r.json()["ok"] is True
    cfg = client.get("/api/voice/config").json()
    assert cfg["voice"] == "leo"
    assert "tara" in cfg["voices"]          # Orpheus voices
    # restore default
    client.post("/api/voice/config", json={"voice": "tara", "model": "gemma4:12b"})
