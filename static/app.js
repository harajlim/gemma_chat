/* ============================================================================
   Gemma 4 — Chat + Voice.  Two toggles give a 2x2 of input x output:

        MIC off / SPEAK off  ->  type  -> text reply (rich chat: tools, markdown)
        MIC off / SPEAK on   ->  type  -> SPOKEN reply   (/api/converse_text)
        MIC on  / SPEAK off  ->  talk  -> text reply     (/api/transcribe -> chat)
        MIC on  / SPEAK on   ->  talk  -> SPOKEN reply    (/api/converse_stream)

   The two SPEAK-on paths are realtime's exact endpoints (Orpheus) — identical
   quality to the realtime_voice app.
   ============================================================================ */

const ORPHEUS_VOICES = ["tara","leah","jess","leo","dan","mia","zac","zoe"];

// ---- shared state ----
let ws = null, sessionId = null;
let pendingImages = [];                 // {id, file, dataUrl}
let modelCapabilities = {}, currentModel = "gemma4:12b";
let isGenerating = false;               // a text/WS chat turn is streaming
let currentBubble = null, currentBubbleText = "";

// ---- input/output toggles ----
let micOn = false;      // voice input (hands-free)
let speakOn = false;    // spoken replies (Orpheus)

// ---- voice runtime (VAD + playback) ----
let running = false;    // mic loop active (== micOn once started)
let turnBusy = false;   // a voice-output turn (fetch->stream->playback) is in flight
let vadState = "idle";  // idle | listening | thinking | speaking
let stream, audioCtx, analyser, dataArr, recorder, chunks = [], mimeType = "";
let vadTimer = null, scheduledSources = [], convAbort = null;

const FRAME_MS = 50, SILENCE_HANG_MS = 500, MIN_SPEECH_MS = 250, MAX_TURN_MS = 20000, CALIBRATE_MS = 400;
let noiseFloor = 0.01, speechThresh = 0.02;

// ---- DOM ----
const chatEl = document.getElementById("chat");
const textarea = document.getElementById("userInput");
const sendBtn = document.getElementById("sendBtn");
const micToggle = document.getElementById("micToggle");
const speakToggle = document.getElementById("speakToggle");
const voiceStatus = document.getElementById("voiceStatus");
const sensEl = document.getElementById("sens");

// ===========================================================================
//  WebSocket chat (text in -> text out, with tools)
// ===========================================================================
function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws/chat`);
  ws.onopen = () => ws.send(JSON.stringify({ session_id: sessionId }));
  ws.onmessage = (e) => handleMessage(JSON.parse(e.data));
  ws.onerror = () => resetTurnState();
  ws.onclose = () => { resetTurnState(); setTimeout(connect, 2000); };
}

function resetTurnState() {
  if (!isGenerating) return;
  isGenerating = false; currentBubble = null; currentBubbleText = "";
  document.getElementById("typing")?.remove();
  sendBtn.disabled = false;
}

function handleMessage(msg) {
  switch (msg.type) {
    case "session": sessionId = msg.session_id; break;
    case "model_set": currentModel = msg.model; document.getElementById("modelSelect").value = msg.model; break;
    case "token":
      if (!currentBubble) { currentBubble = addMessage("assistant", ""); currentBubbleText = ""; }
      currentBubbleText += msg.text;
      renderMarkdown(currentBubble.querySelector(".bubble-content"), currentBubbleText);
      scrollToBottom();
      break;
    case "status": addStatus(msg.text); break;
    case "detection": addDetection(msg); break;
    case "web_search": addWebSearch(msg); break;
    case "stats": if (currentBubble) addStats(currentBubble, msg); updateContextMeter(msg); break;
    case "done":
      isGenerating = false; currentBubble = null; currentBubbleText = ""; sendBtn.disabled = false;
      maybeRearm();
      break;
    case "error": addStatus(msg.text); resetTurnState(); maybeRearm(); break;
  }
}

// ---- rendering ----
function addMessage(role, content, images, opts) {
  opts = opts || {};
  document.getElementById("emptyState")?.remove();
  const div = document.createElement("div");
  div.className = `message ${role}` + (opts.spoken ? " spoken" : "");
  div.innerHTML = `<div class="avatar">${role === "user" ? "U" : "G"}</div><div class="bubble"><div class="bubble-content"></div></div>`;
  const c = div.querySelector(".bubble-content");
  if (images && images.length) for (const img of images) {
    const el = document.createElement("img"); el.src = img.dataUrl || `/images/${img.id}`; el.className = "user-image"; c.appendChild(el);
  }
  if (content) { if (opts.plain) c.textContent = content; else renderMarkdown(c, content); }
  chatEl.appendChild(div); scrollToBottom();
  return div;
}
function addStatus(text) { const d = document.createElement("div"); d.className = "status-msg"; d.textContent = text; chatEl.appendChild(d); scrollToBottom(); }

function appendDetectionCard(contentEl, msg) {
  const card = document.createElement("div"); card.className = "detection-card";
  card.innerHTML = `<div class="det-header"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="1" y="1" width="22" height="22" rx="2"/><line x1="1" y1="8" x2="23" y2="8"/><line x1="8" y1="1" x2="8" y2="23"/></svg> Found ${msg.count} ${escapeHtml(msg.target)}(s)</div><img src="${msg.image_url}" alt="Detection result" loading="lazy"><div class="det-info">Detection time: ${msg.detection_time_s}s</div>`;
  contentEl.appendChild(card); scrollToBottom();
}
function appendSearchCard(contentEl, msg) {
  const card = document.createElement("div"); card.className = "search-card collapsed";
  const rows = (msg.results || []).map(r => `<div class="search-result"><a href="${escapeHtml(safeUrl(r.url))}" target="_blank" rel="noopener noreferrer" class="search-title">${escapeHtml(r.title)}</a><div class="search-url">${escapeHtml(r.url)}</div><div class="search-snippet">${escapeHtml(r.snippet)}</div></div>`).join("");
  card.innerHTML = `<div class="search-header"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg> Web search: "${escapeHtml(msg.query)}" · ${(msg.results||[]).length} results · ${msg.search_time_s}s<svg class="chevron" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 12 15 18 9"/></svg></div><div class="search-body">${rows || '<div class="search-empty">No results</div>'}</div>`;
  card.querySelector(".search-header").addEventListener("click", () => card.classList.toggle("collapsed"));
  contentEl.appendChild(card); scrollToBottom();
}
// chat (WS) path: attach to the current streaming assistant bubble
function addDetection(msg) { if (!currentBubble) { currentBubble = addMessage("assistant", ""); currentBubbleText = ""; } appendDetectionCard(currentBubble.querySelector(".bubble-content"), msg); }
function addWebSearch(msg) { if (!currentBubble) { currentBubble = addMessage("assistant", ""); currentBubbleText = ""; } appendSearchCard(currentBubble.querySelector(".bubble-content"), msg); }
// voice path: each card is its own message in the thread
function voiceDetectionCard(msg) { appendDetectionCard(addMessage("assistant", "").querySelector(".bubble-content"), msg); }
function voiceSearchCard(msg) { appendSearchCard(addMessage("assistant", "").querySelector(".bubble-content"), msg); }
function addStats(messageDiv, stats) {
  const bar = document.createElement("div"); bar.className = "stats-bar"; const items = [];
  if (stats.model) items.push(`<span class="stat">Model: <span class="stat-value">${stats.model}</span></span>`);
  if (stats.total_time_s != null) items.push(`<span class="stat">Total: <span class="stat-value">${stats.total_time_s}s</span></span>`);
  if (stats.ttft_s != null) items.push(`<span class="stat">TTFT: <span class="stat-value">${stats.ttft_s}s</span></span>`);
  if (stats.tokens_per_sec != null) items.push(`<span class="stat">Speed: <span class="stat-value">${stats.tokens_per_sec} tok/s</span></span>`);
  if (stats.eval_tokens != null) items.push(`<span class="stat">Eval: <span class="stat-value">${stats.eval_tokens}</span></span>`);
  bar.innerHTML = items.join(""); messageDiv.querySelector(".bubble").appendChild(bar);
}
const CTX_MAX = 131072;
function updateContextMeter(stats) {
  const used = (stats.prompt_tokens || 0) + (stats.eval_tokens || 0); if (!used) return;
  const pct = Math.min((used / CTX_MAX) * 100, 100), fill = document.getElementById("meterFill");
  fill.style.width = pct + "%"; fill.classList.remove("warn", "danger");
  if (pct > 80) fill.classList.add("danger"); else if (pct > 50) fill.classList.add("warn");
  const fmt = (n) => n >= 1000 ? (n / 1000).toFixed(1) + "K" : n.toString();
  document.getElementById("meterText").textContent = `${fmt(used)} / 131K`;
}
if (window.marked) marked.setOptions({ breaks: true, gfm: true });
function renderMarkdown(el, text) {
  const keep = Array.from(el.querySelectorAll(".user-image, .chat-audio, .detection-card, .search-card"));
  let html; try { html = window.marked ? marked.parse(text || "") : escapeHtml(text || "").replace(/\n/g, "<br>"); }
  catch (e) { html = escapeHtml(text || "").replace(/\n/g, "<br>"); }
  el.innerHTML = html; for (const n of keep) el.appendChild(n);
}
function escapeHtml(s) { return String(s ?? "").replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;").replace(/'/g,"&#39;"); }
function safeUrl(u) { return /^https?:\/\//i.test(u || "") ? u : "#"; }
function scrollToBottom() { chatEl.scrollTop = chatEl.scrollHeight; }

// ===========================================================================
//  Send dispatch — route by the two toggles
// ===========================================================================
async function sendText() {
  const text = textarea.value.trim();
  if (!text && !pendingImages.length) return;
  if (isGenerating || turnBusy) return;
  if (running) pauseListening();          // typing while hands-free: pause the mic for this turn
  if (speakOn) await voiceOutTyped(text); // type -> spoken reply (tools work too)
  else await chatSend(text);              // type -> text reply (rich chat)
}

// Upload any attached images; returns their ids + display info, clears the tray.
async function uploadPending() {
  const imageIds = [], displays = [];
  for (const img of pendingImages) {
    const fd = new FormData(); fd.append("file", img.file);
    try { const d = await (await fetch("/upload", { method: "POST", body: fd })).json(); imageIds.push(d.image_id); displays.push({ id: d.image_id, dataUrl: img.dataUrl }); }
    catch (e) { console.error("upload:", e); }
  }
  pendingImages = []; document.getElementById("imagePreviews").innerHTML = "";
  return { imageIds, displays };
}

// type -> SPOKEN reply (Orpheus), with tools + image context.
async function voiceOutTyped(text) {
  const { imageIds, displays } = await uploadPending();
  const prompt = text || (imageIds.length ? "What's in this image?" : "");
  addMessage("user", text, displays, { plain: true });
  textarea.value = ""; textarea.style.height = "auto";
  await voiceOutTurn(() => {
    const fd = new FormData(); fd.append("text", prompt);
    if (imageIds.length) fd.append("image_ids", imageIds.join(","));
    return fetch("/api/converse_text", { method: "POST", body: fd, signal: convAbort.signal });
  }, { showTranscript: false });
}

// text-out (WS rich chat). `text` may come from typing or from transcribed speech.
async function chatSend(text) {
  if (!ws || ws.readyState !== WebSocket.OPEN) { addStatus("Reconnecting to the server — try again in a moment."); maybeRearm(); return; }
  isGenerating = true; sendBtn.disabled = true;

  const { imageIds, displays: imageDisplays } = await uploadPending();
  addMessage("user", text, imageDisplays);
  textarea.value = ""; textarea.style.height = "auto";

  const typing = document.createElement("div"); typing.className = "message assistant"; typing.id = "typing";
  typing.innerHTML = `<div class="avatar">G</div><div class="bubble"><div class="typing-indicator"><span></span><span></span><span></span></div></div>`;
  chatEl.appendChild(typing); scrollToBottom();

  try { ws.send(JSON.stringify({ text, image_ids: imageIds, audio_ids: [] })); }
  catch (e) { document.getElementById("typing")?.remove(); addStatus("Connection lost — please resend."); textarea.value = text; isGenerating = false; sendBtn.disabled = false; maybeRearm(); return; }

  const orig = ws.onmessage;
  ws.onmessage = (e) => { document.getElementById("typing")?.remove(); ws.onmessage = orig; orig(e); };
}

// ===========================================================================
//  Voice OUTPUT (SPEAK on) — realtime's Orpheus pipeline, verbatim playback
// ===========================================================================
async function voiceOutTurn(makeRequest, { showTranscript }) {
  setState("thinking"); beginTurn();
  convAbort = new AbortController();
  let resp;
  try { resp = await makeRequest(); }
  catch (e) {
    endTurn(); if (convAbort && convAbort.signal.aborted) return;
    addMessage("assistant", "⚠️ Could not reach the local server.", null, { spoken: true, plain: true });
    setState("error"); maybeRearm(); return;
  }
  await playStream(resp, showTranscript);
}

async function playStream(resp, showTranscript) {
  try { await audioCtx?.resume(); } catch (e) {}
  if (!audioCtx) { ensureAudioCtx(); }
  const STARTUP_BUFFER_S = 0.4, pending = [];
  let bufferedSec = 0, started = false, playhead = 0, lastEnd = 0, sawAudio = false, aiBubble = null;
  scheduledSources = [];

  function schedule(int16, sr) {
    if (playhead === 0) playhead = audioCtx.currentTime + 0.06;
    const f32 = new Float32Array(int16.length);
    for (let i = 0; i < int16.length; i++) f32[i] = int16[i] / 32768;
    const ab = audioCtx.createBuffer(1, int16.length, sr); ab.copyToChannel(f32, 0);
    const src = audioCtx.createBufferSource(); src.buffer = ab; src.connect(audioCtx.destination);
    const at = Math.max(playhead, audioCtx.currentTime + 0.02);
    src.start(at); scheduledSources.push(src); playhead = at + ab.duration; lastEnd = playhead;
  }
  function startIfReady(force) {
    if (!started && (force || bufferedSec >= STARTUP_BUFFER_S)) { started = true; setState("speaking"); }
    if (started) while (pending.length) { const a = pending.shift(); schedule(a.int16, a.sr); }
  }

  const reader = resp.body.getReader(), decoder = new TextDecoder(); let buf = "";
  try {
    while (true) {
      const { value, done } = await reader.read(); if (done) break;
      buf += decoder.decode(value, { stream: true });
      let nl;
      while ((nl = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, nl).trim(); buf = buf.slice(nl + 1); if (!line) continue;
        let ev; try { ev = JSON.parse(line); } catch (e) { continue; }
        if (ev.type === "transcript") { if (showTranscript && ev.text) addMessage("user", ev.text, null, { plain: true }); }
        else if (ev.type === "reply_text") { aiBubble = addMessage("assistant", ev.text || "…", null, { spoken: true, plain: true }); }
        else if (ev.type === "audio") { const int16 = b64ToInt16(ev.pcm), sr = ev.sr || 24000; sawAudio = true; pending.push({ int16, sr }); bufferedSec += int16.length / sr; startIfReady(false); }
        else if (ev.type === "web_search") { voiceSearchCard(ev); }
        else if (ev.type === "detection") { voiceDetectionCard(ev); }
        else if (ev.type === "error") { const m = "⚠️ " + (ev.message || "Something went wrong."); if (aiBubble) aiBubble.querySelector(".bubble-content").textContent = m; else addMessage("assistant", m, null, { spoken: true, plain: true }); }
      }
    }
  } catch (e) { if (convAbort && convAbort.signal.aborted) { endTurn(); return; } }
  if (convAbort && convAbort.signal.aborted) { endTurn(); return; }
  startIfReady(true);
  const finish = () => { endTurn(); maybeRearm(); if (!micOn) setState("idle"); };
  if (!sawAudio) { finish(); return; }
  const waitMs = Math.max(0, (lastEnd - audioCtx.currentTime) * 1000) + 150;
  setTimeout(finish, waitMs);
}
function b64ToInt16(b64) { const bin = atob(b64), u8 = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i); return new Int16Array(u8.buffer, 0, u8.length >> 1); }

// ===========================================================================
//  Voice INPUT (MIC on) — hands-free VAD; routes by the SPEAK toggle
// ===========================================================================
function setState(s) {
  vadState = s;
  micToggle.classList.toggle("listening", s === "listening");
  voiceStatus.textContent = s === "listening" ? "Listening…" : s === "thinking" ? "Thinking…" : s === "speaking" ? "Speaking…" : "";
}
function beginTurn() { turnBusy = true; sendBtn.disabled = true; textarea.disabled = true; }
function endTurn() { turnBusy = false; sendBtn.disabled = isGenerating; textarea.disabled = false; }

function sensitivity() { const v = parseInt(sensEl.value, 10); return 3.2 - (v - 1) * 0.22; }
function pickMime() { for (const o of ["audio/webm;codecs=opus", "audio/webm", "audio/mp4", "audio/ogg"]) if (MediaRecorder.isTypeSupported(o)) return o; return ""; }
function ensureAudioCtx() { if (!audioCtx || audioCtx.state === "closed") audioCtx = new (window.AudioContext || window.webkitAudioContext)(); return audioCtx; }
function unlockAudio() { ensureAudioCtx(); if (audioCtx.state === "suspended") audioCtx.resume(); try { const b = audioCtx.createBuffer(1, 1, 22050); const s = audioCtx.createBufferSource(); s.buffer = b; s.connect(audioCtx.destination); s.start(0); } catch (e) {} }

async function startConversation() {
  running = true;
  stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true } });
  ensureAudioCtx();
  const src = audioCtx.createMediaStreamSource(stream);
  analyser = audioCtx.createAnalyser(); analyser.fftSize = 1024; dataArr = new Float32Array(analyser.fftSize); src.connect(analyser);
  mimeType = pickMime();
  if (!isGenerating && !turnBusy) armListening(); else setState("thinking");
}
// Turning the mic OFF stops voice INPUT only. Any reply already in flight keeps
// playing AND still commits to history — flipping the mic must not kill the answer
// (killing it mid-turn drops the turn from history and corrupts later context).
function stopConversation() {
  running = false;
  clearInterval(vadTimer);
  if (recorder && recorder.state !== "inactive") { recorder.onstop = null; recorder.stop(); }
  if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
  if (!turnBusy) setState("idle");   // if a turn is speaking, let it finish -> idle on its own
}

// Hard stop: abort the in-flight turn AND kill playback. Used by Clear chat only.
function abortVoice() {
  if (convAbort) { try { convAbort.abort(); } catch (e) {} convAbort = null; }
  scheduledSources.forEach(s => { try { s.stop(); } catch (e) {} });
  scheduledSources = [];
  endTurn();
}
function pauseListening() { clearInterval(vadTimer); if (recorder && recorder.state !== "inactive") { recorder.onstop = null; recorder.stop(); } }
function maybeRearm() { if (micOn && running && !turnBusy && !isGenerating) armListening(); }

function rms() { analyser.getFloatTimeDomainData(dataArr); let s = 0; for (let i = 0; i < dataArr.length; i++) s += dataArr[i] * dataArr[i]; return Math.sqrt(s / dataArr.length); }

function armListening() {
  if (!running) return;
  setState("listening"); chunks = [];
  recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
  recorder.ondataavailable = e => { if (e.data.size) chunks.push(e.data); };
  recorder.start(100);
  let calibrating = true, calStart = performance.now(), calSum = 0, calN = 0;
  let speechStart = null, lastVoice = null, turnStart = performance.now();
  clearInterval(vadTimer);
  vadTimer = setInterval(() => {
    if (!running || vadState !== "listening") return;
    const level = rms(), now = performance.now();
    if (calibrating) { calSum += level; calN++; if (now - calStart >= CALIBRATE_MS) { noiseFloor = Math.max(0.006, calSum / Math.max(1, calN)); speechThresh = noiseFloor * sensitivity() + 0.004; calibrating = false; } return; }
    const voiced = level > speechThresh;
    if (voiced) { if (speechStart === null) speechStart = now; lastVoice = now; }
    const spokeEnough = speechStart !== null && (lastVoice - speechStart) > MIN_SPEECH_MS;
    const wentQuiet = lastVoice !== null && (now - lastVoice) > SILENCE_HANG_MS;
    if (spokeEnough && wentQuiet) { finalizeTurn(); return; }
    if (now - turnStart > MAX_TURN_MS && speechStart !== null) { finalizeTurn(); return; }
  }, FRAME_MS);
}
function finalizeTurn() { clearInterval(vadTimer); setState("thinking"); recorder.onstop = sendClip; if (recorder.state !== "inactive") recorder.stop(); }

async function sendClip() {
  const blob = new Blob(chunks, { type: mimeType || "audio/webm" });
  if (blob.size < 1200) { armListening(); return; }       // basically silence
  const ext = mimeType.includes("mp4") ? "mp4" : (mimeType.includes("ogg") ? "ogg" : "webm");

  if (speakOn) {
    // talk -> SPOKEN reply (realtime /api/converse_stream); pass any attached image
    const { imageIds } = await uploadPending();
    await voiceOutTurn(() => { const fd = new FormData(); fd.append("audio", blob, "turn." + ext); if (imageIds.length) fd.append("image_ids", imageIds.join(",")); return fetch("/api/converse_stream", { method: "POST", body: fd, signal: convAbort.signal }); }, { showTranscript: true });
  } else {
    // talk -> TEXT reply: transcribe, then feed the rich chat
    setState("thinking");
    let text = "";
    try { const fd = new FormData(); fd.append("audio", blob, "turn." + ext); text = (await (await fetch("/api/transcribe", { method: "POST", body: fd })).json()).text || ""; }
    catch (e) { addStatus("Transcription failed."); maybeRearm(); return; }
    if (!text.trim()) { armListening(); return; }
    setState("");
    await chatSend(text);     // re-arm happens on the WS 'done'
  }
}

// ===========================================================================
//  Images (rich chat only) — file / drag / paste
// ===========================================================================
document.getElementById("fileInput").addEventListener("change", (e) => { for (const f of e.target.files) addPendingImage(f); e.target.value = ""; });
function addPendingImage(file) { const r = new FileReader(); r.onload = (e) => { pendingImages.push({ id: Math.random().toString(36).slice(2, 10), file, dataUrl: e.target.result }); renderPreviews(); }; r.readAsDataURL(file); }
function removePendingImage(id) { pendingImages = pendingImages.filter(i => i.id !== id); renderPreviews(); }
function renderPreviews() { document.getElementById("imagePreviews").innerHTML = pendingImages.map(i => `<div class="img-preview"><img src="${i.dataUrl}"><button class="remove-btn" onclick="removePendingImage('${i.id}')">&times;</button></div>`).join(""); }
let dragCounter = 0;
document.addEventListener("dragenter", (e) => { e.preventDefault(); dragCounter++; document.getElementById("dragOverlay").classList.add("active"); });
document.addEventListener("dragleave", (e) => { e.preventDefault(); if (--dragCounter === 0) document.getElementById("dragOverlay").classList.remove("active"); });
document.addEventListener("dragover", (e) => e.preventDefault());
document.addEventListener("drop", (e) => { e.preventDefault(); dragCounter = 0; document.getElementById("dragOverlay").classList.remove("active"); for (const f of e.dataTransfer.files) if (f.type.startsWith("image/")) addPendingImage(f); });
textarea.addEventListener("input", () => { textarea.style.height = "auto"; textarea.style.height = Math.min(textarea.scrollHeight, 160) + "px"; });
textarea.addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendText(); } });
textarea.addEventListener("paste", (e) => { const items = e.clipboardData?.items; if (!items) return; for (const it of items) if (it.type.startsWith("image/")) { e.preventDefault(); const f = it.getAsFile(); if (f) addPendingImage(f); } });

// ===========================================================================
//  Clear
// ===========================================================================
function clearChat() {
  abortVoice();                       // full reset: kill any in-flight reply + playback
  if (running) { micOn = false; updateMicUI(); stopConversation(); }
  sessionId = null; currentBubble = null; currentBubbleText = ""; isGenerating = false;
  pendingImages = []; document.getElementById("imagePreviews").innerHTML = "";
  chatEl.innerHTML = `<div class="empty-state" id="emptyState"><h2>Gemma 4 — Chat + Voice</h2><p>Type for rich chat with web search, object detection and images. Flip <b>Mic</b> to talk hands-free, and <b>Speak</b> to hear replies aloud. Everything runs locally.</p></div>`;
  document.getElementById("meterFill").style.width = "0%"; document.getElementById("meterFill").classList.remove("warn", "danger");
  document.getElementById("meterText").textContent = "0 / 131K";
  fetch("/api/voice/reset", { method: "POST" }).catch(() => {});
  if (ws) { ws.onopen = ws.onmessage = ws.onerror = ws.onclose = null; try { ws.close(); } catch (e) {} }
  connect(); sendBtn.disabled = false;
}

// ===========================================================================
//  Model picker (shared brain)
// ===========================================================================
async function loadModels() {
  try {
    const data = await (await fetch("/models")).json();
    modelCapabilities = data.models;
    const sel = document.getElementById("modelSelect"); sel.innerHTML = "";
    for (const m of Object.keys(data.models)) { const o = document.createElement("option"); o.value = m; o.textContent = m; if (m === data.default) o.selected = true; sel.appendChild(o); }
    currentModel = data.default;
  } catch (e) { console.error("loadModels:", e); }
}
document.getElementById("modelSelect").addEventListener("change", (e) => { currentModel = e.target.value; if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: "set_model", model: currentModel })); });

// ===========================================================================
//  Toggles
// ===========================================================================
function updateMicUI() {
  micToggle.classList.toggle("on", micOn);
  micToggle.setAttribute("aria-pressed", micOn ? "true" : "false");
  document.getElementById("micState").textContent = micOn ? "on" : "off";
  if (!micOn) { voiceStatus.textContent = ""; micToggle.classList.remove("listening"); }
}
function updateSpeakUI() {
  speakToggle.classList.toggle("on", speakOn);
  speakToggle.setAttribute("aria-pressed", speakOn ? "true" : "false");
  document.getElementById("speakState").textContent = speakOn ? "on" : "off";
}
micToggle.addEventListener("click", async () => {
  unlockAudio();
  micOn = !micOn; updateMicUI();
  if (micOn) {
    try { await startConversation(); }
    catch (e) { micOn = false; updateMicUI(); voiceStatus.textContent = "Mic blocked — allow microphone access."; }
  } else { stopConversation(); }
});
speakToggle.addEventListener("click", () => { unlockAudio(); speakOn = !speakOn; updateSpeakUI(); });

// ===========================================================================
//  Settings (voice + personality + sensitivity)
// ===========================================================================
let defaultPrompt = "";
const voiceSel = document.getElementById("voiceSel");
async function loadConfig() {
  try {
    const c = await (await fetch("/api/voice/config")).json();
    defaultPrompt = c.default_system_prompt || "";
    voiceSel.innerHTML = "";
    (c.voices || ORPHEUS_VOICES).forEach(v => { const o = document.createElement("option"); o.value = v; o.textContent = v; if (v === c.voice) o.selected = true; voiceSel.appendChild(o); });
    document.getElementById("sysprompt").value = c.system_prompt || "";
  } catch (e) {}
}
async function applyConfig() {
  const st = document.getElementById("cfgStatus"); st.textContent = "saving…";
  try {
    await fetch("/api/voice/config", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ voice: voiceSel.value, system_prompt: document.getElementById("sysprompt").value }) });
    st.textContent = "✓ applied — next spoken turn"; setTimeout(() => { st.textContent = ""; }, 2500);
  } catch (e) { st.textContent = "⚠️ failed"; }
}
document.getElementById("applyCfg").addEventListener("click", applyConfig);
document.getElementById("resetCfg").addEventListener("click", () => { document.getElementById("sysprompt").value = defaultPrompt; applyConfig(); });
document.addEventListener("click", (e) => { const s = document.getElementById("settings"); if (s.open && !s.contains(e.target)) s.removeAttribute("open"); });

// ---- init ----
updateMicUI(); updateSpeakUI();
loadModels(); connect(); loadConfig();
textarea.focus();
