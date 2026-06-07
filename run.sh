#!/bin/bash
# ---------------------------------------------------------------------------
# Launch Gemma 4 Chat + Voice.
#
#   ./run.sh                 # localhost only        -> http://localhost:8000
#   ./run.sh --tailnet       # + this machine's Tailscale IP (phone/tailnet only,
#                            #   NOT the wider local network)
#   PORT=8010 ./run.sh       # different port
#   VENV_PYTHON=/path/python ./run.sh   # use a specific interpreter
#
# Tailnet mode binds 127.0.0.1 AND the 100.x Tailscale IP — never 0.0.0.0 — so
# only this laptop and your privileged tailnet devices can reach it.
# ---------------------------------------------------------------------------
cd "$(dirname "$0")"

# --- pick the Python interpreter ------------------------------------------
# The merged app needs BOTH stacks: ollama/Pillow/ddgs (chat+tools) and the ML
# stack mlx_whisper/kokoro_onnx/snac/torch (voice). The real_time_voice .venv has
# the ML stack; we added the chat deps to it. Override with VENV_PYTHON if you
# built your own combined env (see requirements.txt).
DEFAULT_VENV="/Users/mharajli/Desktop/agent_space/real_time_voice/.venv/bin/python"
PY="${VENV_PYTHON:-$DEFAULT_VENV}"
if [ ! -x "$PY" ]; then
  if [ -x "venv/bin/python" ]; then PY="venv/bin/python"; else PY="python3"; fi
  echo "[run] (note) voice .venv not found; using $PY (voice features need the ML stack — see requirements.txt)"
fi

# --- sanity: is Ollama up? ------------------------------------------------
curl -s http://localhost:11434/api/tags >/dev/null \
  || echo "[run] (warning) Ollama not reachable on :11434 — start the Ollama app first."

# --- models are cached locally; forbid runtime network for the ML libs -----
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_DISABLE_TELEMETRY=1

export PORT="${PORT:-8000}"

# --- tailnet flag ----------------------------------------------------------
WANT_TAILNET=0
for a in "$@"; do [ "$a" = "--tailnet" ] && WANT_TAILNET=1; done
[ "${TAILNET:-}" = "1" ] && WANT_TAILNET=1

export BIND_HOSTS="127.0.0.1"
if [ "$WANT_TAILNET" = "1" ]; then
  TS="tailscale"
  command -v tailscale >/dev/null 2>&1 || TS="/Applications/Tailscale.app/Contents/MacOS/Tailscale"
  TSIP=$("$TS" ip -4 2>/dev/null | head -1)
  if [ -n "$TSIP" ]; then
    export BIND_HOSTS="127.0.0.1,$TSIP"
    echo "[run] tailnet mode: also on  ->  http://$TSIP:$PORT   (tailnet only — NOT the local network)"
    echo "[run]   From your phone on the same tailnet, open that URL."
    echo "[run]   • Typing + hearing replies works over plain http."
    echo "[run]   • The MIC needs HTTPS (browsers block getUserMedia off-secure-context)."
    echo "[run]     To talk from your phone: run 'tailscale serve $PORT' and open the https URL it prints."
  else
    echo "[run] (warning) --tailnet set but no Tailscale IP found (is Tailscale running?). Localhost only."
  fi
fi

echo "[run] Gemma 4 Chat + Voice on  ->  http://localhost:$PORT   (Ctrl-C to stop)"
exec "$PY" server.py
