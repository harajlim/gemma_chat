// Voice playground — type text (with emotion tags), stream + play it via /api/speak.
const ta = document.getElementById("text");
const voiceSel = document.getElementById("voice");
const speakBtn = document.getElementById("speak");
const stopBtn = document.getElementById("stop");
const statusEl = document.getElementById("status");
const tagsEl = document.getElementById("tags");

const TAGS = ["<laugh>", "<chuckle>", "<giggle>", "<sigh>", "<gasp>", "<yawn>", "<groan>", "<sniffle>", "<cough>"];
TAGS.forEach(t => {
  const b = document.createElement("button");
  b.className = "tag"; b.textContent = t;
  b.onclick = () => insertAtCursor(" " + t + " ");
  tagsEl.appendChild(b);
});

function insertAtCursor(s) {
  const start = ta.selectionStart, end = ta.selectionEnd;
  ta.value = ta.value.slice(0, start) + s + ta.value.slice(end);
  ta.selectionStart = ta.selectionEnd = start + s.length;
  ta.focus();
}

let actx = null, sources = [], abort = null;

function b64ToInt16(b64) {
  const bin = atob(b64), u8 = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  return new Int16Array(u8.buffer, 0, u8.length >> 1);
}

function setBusy(b) {
  speakBtn.disabled = b;
  statusEl.textContent = b ? "speaking…" : "Type something and press Speak.";
}

function stop() {
  if (abort) { try { abort.abort(); } catch (e) {} abort = null; }
  sources.forEach(s => { try { s.stop(); } catch (e) {} });
  sources = [];
  setBusy(false);
}

async function speak() {
  const text = ta.value.trim();
  if (!text) return;
  stop();
  actx = actx || new (window.AudioContext || window.webkitAudioContext)();
  try { await actx.resume(); } catch (e) {}
  abort = new AbortController();
  setBusy(true);

  const STARTUP_BUFFER_S = 0.4;
  const pending = [];
  let bufferedSec = 0, started = false, playhead = 0, lastEnd = 0, sawAudio = false;

  function schedule(int16, sr) {
    if (playhead === 0) playhead = actx.currentTime + 0.06;
    const f32 = new Float32Array(int16.length);
    for (let i = 0; i < int16.length; i++) f32[i] = int16[i] / 32768;
    const ab = actx.createBuffer(1, int16.length, sr); ab.copyToChannel(f32, 0);
    const src = actx.createBufferSource(); src.buffer = ab; src.connect(actx.destination);
    const at = Math.max(playhead, actx.currentTime + 0.02);
    src.start(at); sources.push(src);
    playhead = at + ab.duration; lastEnd = playhead;
  }
  function startIfReady(force) {
    if (!started && (force || bufferedSec >= STARTUP_BUFFER_S)) started = true;
    if (started) while (pending.length) { const a = pending.shift(); schedule(a.int16, a.sr); }
  }

  let resp;
  try {
    const fd = new FormData();
    fd.append("text", text);
    fd.append("voice", voiceSel.value);
    resp = await fetch("/api/speak", { method: "POST", body: fd, signal: abort.signal });
  } catch (e) {
    if (abort) statusEl.textContent = "⚠️ server unreachable";
    setBusy(false); return;
  }

  const reader = resp.body.getReader(), decoder = new TextDecoder();
  let buf = "";
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let nl;
      while ((nl = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, nl).trim(); buf = buf.slice(nl + 1);
        if (!line) continue;
        let ev; try { ev = JSON.parse(line); } catch (e) { continue; }
        if (ev.type === "audio") {
          const int16 = b64ToInt16(ev.pcm), sr = ev.sr || 24000;
          sawAudio = true; pending.push({ int16, sr });
          bufferedSec += int16.length / sr; startIfReady(false);
        } else if (ev.type === "error") {
          statusEl.textContent = "⚠️ " + (ev.message || "synthesis failed");
        }
      }
    }
  } catch (e) {
    if (!abort) return;
  }
  startIfReady(true);
  const waitMs = sawAudio ? Math.max(0, (lastEnd - actx.currentTime) * 1000) + 150 : 0;
  setTimeout(() => { speakBtn.disabled = false; statusEl.textContent = "done — press Speak again"; }, waitMs);
}

speakBtn.addEventListener("click", speak);
stopBtn.addEventListener("click", stop);
ta.addEventListener("keydown", (e) => { if ((e.metaKey || e.ctrlKey) && e.key === "Enter") speak(); });
