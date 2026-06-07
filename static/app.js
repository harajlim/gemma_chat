/* ============================================================================
   Gemma 4 — Chat + Voice front-end.

   Two input paths into ONE conversation thread:
     • TEXT  — WebSocket /ws/chat: streaming tokens, markdown, image upload,
               tool calling (web_search + detect_objects), live stats.
     • VOICE — hands-free VAD orb -> /api/converse_stream (NDJSON): Whisper STT,
               the brain, streamed TTS played gaplessly. Behaviour-identical to
               the original real_time_voice app.
   ============================================================================ */

// ----- voices (mirror voice/engine.py) -----
const KOKORO_VOICES = ["af_heart","af_bella","af_nicole","af_sarah","af_sky",
  "am_adam","am_michael","am_fenrir","bf_emma","bf_isabella","bm_george","bm_lewis"];
const ORPHEUS_VOICES = ["tara","leah","jess","leo","dan","mia","zac","zoe"];

// ===========================================================================
//  Shared state
// ===========================================================================
let ws = null;
let sessionId = null;
let pendingImages = [];        // {id, file, dataUrl}
let pendingAudios = [];        // {id, file, blobUrl}  — Gemma native audio attach (e4b/e2b)
let modelCapabilities = {};    // model -> {vision, audio, label}
let currentModel = "gemma4:12b";
let isGenerating = false;      // a TEXT (WS) turn is streaming
let currentBubble = null, currentBubbleText = "";
let chatRecorder = null, isChatRecording = false;

// DOM
const chatEl   = document.getElementById("chat");
const orb      = document.getElementById("orb");
const orbLabel = document.getElementById("orbLabel");
const voiceHint = document.getElementById("voiceHint");
const sensEl   = document.getElementById("sens");
const textarea = document.getElementById("userInput");
const sendBtn  = document.getElementById("sendBtn");

// ===========================================================================
//  WebSocket text chat
// ===========================================================================
function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws/chat`);
  ws.onopen = () => ws.send(JSON.stringify({ session_id: sessionId }));
  ws.onmessage = (e) => handleMessage(JSON.parse(e.data));
  // If the socket drops mid-turn, the `done` frame never arrives — self-heal so
  // a stuck text turn can NEVER leave the voice orb disabled (it's gated on
  // isGenerating). Both onclose and onerror reset, then we reconnect.
  ws.onerror = () => resetTurnState();
  ws.onclose = () => { resetTurnState(); setTimeout(connect, 2000); };
}

// Clear any in-flight TEXT turn state. Safe to call repeatedly. Crucially this
// re-enables the voice orb (see updateControls) so the two paths can't deadlock.
function resetTurnState() {
  if (!isGenerating) return;
  isGenerating = false;
  currentBubble = null;
  currentBubbleText = "";
  document.getElementById("typing")?.remove();
  updateControls();
}

function handleMessage(msg) {
  switch (msg.type) {
    case "session": sessionId = msg.session_id; break;
    case "model_set":
      currentModel = msg.model;
      document.getElementById("modelSelect").value = msg.model;
      updateChatMicState();
      break;
    case "token":
      if (!currentBubble) { currentBubble = addMessage("assistant", ""); currentBubbleText = ""; }
      currentBubbleText += msg.text;
      renderMarkdown(currentBubble.querySelector(".bubble-content"), currentBubbleText);
      scrollToBottom();
      break;
    case "status": addStatus(msg.text); break;
    case "detection": addDetection(msg); break;
    case "web_search": addWebSearch(msg); break;
    case "stats":
      if (currentBubble) addStats(currentBubble, msg);
      updateContextMeter(msg);
      break;
    case "done":
      isGenerating = false; currentBubble = null; currentBubbleText = "";
      updateControls();
      break;
    case "error": addStatus(msg.text); resetTurnState(); break;
  }
}

// ----- rendering -----
function addMessage(role, content, images, opts) {
  document.getElementById("emptyState")?.remove();
  const div = document.createElement("div");
  div.className = `message ${role}` + (opts && opts.voice ? " voice" : "");
  const avatarLabel = role === "user" ? "U" : "G";
  div.innerHTML = `<div class="avatar">${avatarLabel}</div><div class="bubble"><div class="bubble-content"></div></div>`;
  const content_el = div.querySelector(".bubble-content");

  if (images && images.length) {
    for (const img of images) {
      const el = document.createElement("img");
      el.src = img.dataUrl || `/images/${img.id}`;
      el.className = "user-image";
      content_el.appendChild(el);
    }
  }
  if (opts && opts.audios && opts.audios.length) {
    for (const aud of opts.audios) {
      const el = document.createElement("audio");
      el.src = aud.blobUrl || `/audio/${aud.id}`;
      el.controls = true; el.className = "chat-audio";
      content_el.appendChild(el);
    }
  }
  if (content) {
    if (opts && opts.plain) content_el.textContent = content;
    else renderMarkdown(content_el, content);
  }
  chatEl.appendChild(div);
  scrollToBottom();
  return div;
}

function addStatus(text) {
  const div = document.createElement("div");
  div.className = "status-msg";
  div.textContent = text;
  chatEl.appendChild(div);
  scrollToBottom();
}

function addDetection(msg) {
  if (!currentBubble) { currentBubble = addMessage("assistant", ""); currentBubbleText = ""; }
  const card = document.createElement("div");
  card.className = "detection-card";
  card.innerHTML = `
    <div class="det-header">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="1" y="1" width="22" height="22" rx="2"/><line x1="1" y1="8" x2="23" y2="8"/><line x1="8" y1="1" x2="8" y2="23"/></svg>
      Found ${msg.count} ${escapeHtml(msg.target)}(s)
    </div>
    <img src="${msg.image_url}" alt="Detection result" loading="lazy">
    <div class="det-info">Detection time: ${msg.detection_time_s}s</div>`;
  currentBubble.querySelector(".bubble-content").appendChild(card);
  scrollToBottom();
}

function addWebSearch(msg) {
  if (!currentBubble) { currentBubble = addMessage("assistant", ""); currentBubbleText = ""; }
  const card = document.createElement("div");
  card.className = "search-card collapsed";
  const resultsHtml = (msg.results || []).map(r => `
    <div class="search-result">
      <a href="${escapeHtml(safeUrl(r.url))}" target="_blank" rel="noopener noreferrer" class="search-title">${escapeHtml(r.title)}</a>
      <div class="search-url">${escapeHtml(r.url)}</div>
      <div class="search-snippet">${escapeHtml(r.snippet)}</div>
    </div>`).join("");
  const count = (msg.results || []).length;
  card.innerHTML = `
    <div class="search-header">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
      Web search: "${escapeHtml(msg.query)}" · ${count} results · ${msg.search_time_s}s
      <svg class="chevron" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 12 15 18 9"/></svg>
    </div>
    <div class="search-body">${resultsHtml || '<div class="search-empty">No results</div>'}</div>`;
  card.querySelector(".search-header").addEventListener("click", () => card.classList.toggle("collapsed"));
  currentBubble.querySelector(".bubble-content").appendChild(card);
  scrollToBottom();
}

function addStats(messageDiv, stats) {
  const bubble = messageDiv.querySelector(".bubble");
  const bar = document.createElement("div");
  bar.className = "stats-bar";
  const items = [];
  if (stats.model) items.push(`<span class="stat">Model: <span class="stat-value">${stats.model}</span></span>`);
  if (stats.total_time_s != null) items.push(`<span class="stat">Total: <span class="stat-value">${stats.total_time_s}s</span></span>`);
  if (stats.ttft_s != null) items.push(`<span class="stat">TTFT: <span class="stat-value">${stats.ttft_s}s</span></span>`);
  if (stats.tokens_per_sec != null) items.push(`<span class="stat">Speed: <span class="stat-value">${stats.tokens_per_sec} tok/s</span></span>`);
  if (stats.approx_tokens != null) items.push(`<span class="stat">Tokens: <span class="stat-value">~${stats.approx_tokens}</span></span>`);
  if (stats.stream_time_s != null) items.push(`<span class="stat">Gen: <span class="stat-value">${stats.stream_time_s}s</span></span>`);
  if (stats.eval_tokens != null) items.push(`<span class="stat">Eval: <span class="stat-value">${stats.eval_tokens}</span></span>`);
  if (stats.prompt_tokens != null) items.push(`<span class="stat">Prompt: <span class="stat-value">${stats.prompt_tokens}</span></span>`);
  bar.innerHTML = items.join("");
  bubble.appendChild(bar);
}

const CTX_MAX = 131072;
function updateContextMeter(stats) {
  const used = (stats.prompt_tokens || 0) + (stats.eval_tokens || 0);
  if (!used) return;
  const pct = Math.min((used / CTX_MAX) * 100, 100);
  const fill = document.getElementById("meterFill");
  fill.style.width = pct + "%";
  fill.classList.remove("warn", "danger");
  if (pct > 80) fill.classList.add("danger");
  else if (pct > 50) fill.classList.add("warn");
  const fmt = (n) => n >= 1000 ? (n / 1000).toFixed(1) + "K" : n.toString();
  document.getElementById("meterText").textContent = `${fmt(used)} / 131K`;
}

if (window.marked) marked.setOptions({ breaks: true, gfm: true });

function renderMarkdown(el, text) {
  const preserved = Array.from(el.querySelectorAll(".user-image, .chat-audio, .detection-card, .search-card"));
  let html;
  try { html = window.marked ? marked.parse(text || "") : escapeHtml(text || "").replace(/\n/g, "<br>"); }
  catch (e) { html = escapeHtml(text || "").replace(/\n/g, "<br>"); }
  el.innerHTML = html;
  for (const node of preserved) el.appendChild(node);
}

function escapeHtml(s) {
  return String(s ?? "")
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}
// Only allow http(s) links from third-party (web-search) content — never javascript: etc.
function safeUrl(u) { return /^https?:\/\//i.test(u || "") ? u : "#"; }
function scrollToBottom() { chatEl.scrollTop = chatEl.scrollHeight; }

// ----- sending a text turn -----
async function sendMessage() {
  const text = textarea.value.trim();
  if (!text && !pendingImages.length && !pendingAudios.length) return;
  if (isGenerating) return;
  if (running) return;                 // don't cross-talk with an active voice turn
  if (!ws || ws.readyState !== WebSocket.OPEN) {
    addStatus("Reconnecting to the server — try again in a moment.");
    return;
  }
  if (isChatRecording) stopChatRecording();

  isGenerating = true; updateControls();

  const imageIds = [], imageDisplays = [];
  for (const img of pendingImages) {
    const fd = new FormData(); fd.append("file", img.file);
    try {
      const data = await (await fetch("/upload", { method: "POST", body: fd })).json();
      imageIds.push(data.image_id);
      imageDisplays.push({ id: data.image_id, dataUrl: img.dataUrl });
    } catch (e) { console.error("Upload failed:", e); }
  }
  const audioIds = [], audioDisplays = [];
  for (const aud of pendingAudios) {
    const fd = new FormData(); fd.append("file", aud.file);
    try {
      const data = await (await fetch("/upload_audio", { method: "POST", body: fd })).json();
      audioIds.push(data.audio_id);
      audioDisplays.push({ id: data.audio_id, blobUrl: aud.blobUrl });
    } catch (e) { console.error("Audio upload failed:", e); }
  }

  addMessage("user", text, imageDisplays, { audios: audioDisplays });
  textarea.value = ""; textarea.style.height = "auto";
  pendingImages = []; document.getElementById("imagePreviews").innerHTML = "";
  pendingAudios = []; document.getElementById("audioPreviews").innerHTML = "";

  // typing indicator
  const typing = document.createElement("div");
  typing.className = "message assistant"; typing.id = "typing";
  typing.innerHTML = `<div class="avatar">G</div><div class="bubble"><div class="typing-indicator"><span></span><span></span><span></span></div></div>`;
  chatEl.appendChild(typing); scrollToBottom();

  try {
    ws.send(JSON.stringify({ text, image_ids: imageIds, audio_ids: audioIds }));
  } catch (e) {
    // Socket dropped between the readyState check and now — roll back so nothing wedges.
    document.getElementById("typing")?.remove();
    addStatus("Connection lost — please resend.");
    textarea.value = text;
    isGenerating = false; currentBubble = null; updateControls();
    return;
  }

  const orig = ws.onmessage;
  ws.onmessage = (e) => { document.getElementById("typing")?.remove(); ws.onmessage = orig; orig(e); };
}

function updateControls() {
  sendBtn.disabled = isGenerating || turnBusy;
  // orb is disabled mid text-generation so the two paths never overlap
  orb.classList.toggle("disabled", isGenerating);
  orb.style.pointerEvents = isGenerating ? "none" : "";
  orb.style.opacity = isGenerating ? ".4" : "";
  orb.setAttribute("aria-disabled", isGenerating ? "true" : "false");
  orb.tabIndex = isGenerating ? -1 : 0;
  const cm = document.getElementById("chatMicBtn");
  if (cm) cm.disabled = isGenerating || running;
}

// ----- images: file picker / drag / paste -----
document.getElementById("fileInput").addEventListener("change", (e) => {
  for (const file of e.target.files) addPendingImage(file);
  e.target.value = "";
});
function addPendingImage(file) {
  const reader = new FileReader();
  reader.onload = (e) => {
    pendingImages.push({ id: Math.random().toString(36).slice(2, 10), file, dataUrl: e.target.result });
    renderPreviews();
  };
  reader.readAsDataURL(file);
}
function removePendingImage(id) { pendingImages = pendingImages.filter(i => i.id !== id); renderPreviews(); }
function renderPreviews() {
  document.getElementById("imagePreviews").innerHTML = pendingImages.map(img => `
    <div class="img-preview"><img src="${img.dataUrl}"><button class="remove-btn" onclick="removePendingImage('${img.id}')">&times;</button></div>`).join("");
}

// ----- Gemma native audio attach (record a clip for the model to "listen" to;
//        e4b/e2b only). Kept deliberately SEPARATE from the realtime voice orb. -----
function updateChatMicState() {
  const btn = document.getElementById("chatMicBtn");
  if (!btn) return;
  const caps = modelCapabilities[currentModel];
  const supported = !!(caps && caps.audio);
  btn.hidden = !supported;          // hidden entirely for non-audio brains (e.g. the 12b default)
  btn.title = supported ? "Record an audio clip for the model" : "";
  if (!supported && isChatRecording) stopChatRecording();
}

function toggleChatRecording() { isChatRecording ? stopChatRecording() : startChatRecording(); }

async function startChatRecording() {
  if (running || isGenerating) return;
  try {
    const micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const chunks = [];
    chatRecorder = new MediaRecorder(micStream, { mimeType: "audio/webm" });
    chatRecorder.ondataavailable = (e) => { if (e.data.size > 0) chunks.push(e.data); };
    chatRecorder.onstop = () => {
      micStream.getTracks().forEach(t => t.stop());
      const blob = new Blob(chunks, { type: "audio/webm" });
      const file = new File([blob], `clip_${Date.now()}.webm`, { type: "audio/webm" });
      pendingAudios.push({ id: Math.random().toString(36).slice(2, 10), file, blobUrl: URL.createObjectURL(blob) });
      renderAudioPreviews();
    };
    chatRecorder.start();
    isChatRecording = true;
    document.getElementById("chatMicBtn").classList.add("recording");
  } catch (e) { console.error("Mic access denied:", e); }
}

function stopChatRecording() {
  if (chatRecorder && chatRecorder.state !== "inactive") chatRecorder.stop();
  isChatRecording = false;
  document.getElementById("chatMicBtn")?.classList.remove("recording");
}

function removePendingAudio(id) {
  const a = pendingAudios.find(x => x.id === id);
  if (a) URL.revokeObjectURL(a.blobUrl);
  pendingAudios = pendingAudios.filter(x => x.id !== id);
  renderAudioPreviews();
}

function renderAudioPreviews() {
  document.getElementById("audioPreviews").innerHTML = pendingAudios.map(a => `
    <div class="audio-preview"><audio src="${a.blobUrl}" controls></audio>
      <button class="remove-btn" onclick="removePendingAudio('${a.id}')">&times;</button></div>`).join("");
}

let dragCounter = 0;
document.addEventListener("dragenter", (e) => { e.preventDefault(); dragCounter++; document.getElementById("dragOverlay").classList.add("active"); });
document.addEventListener("dragleave", (e) => { e.preventDefault(); if (--dragCounter === 0) document.getElementById("dragOverlay").classList.remove("active"); });
document.addEventListener("dragover", (e) => e.preventDefault());
document.addEventListener("drop", (e) => {
  e.preventDefault(); dragCounter = 0; document.getElementById("dragOverlay").classList.remove("active");
  for (const file of e.dataTransfer.files) if (file.type.startsWith("image/")) addPendingImage(file);
});

textarea.addEventListener("input", () => { textarea.style.height = "auto"; textarea.style.height = Math.min(textarea.scrollHeight, 160) + "px"; });
textarea.addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); } });
textarea.addEventListener("paste", (e) => {
  const items = e.clipboardData?.items; if (!items) return;
  for (const item of items) if (item.type.startsWith("image/")) { e.preventDefault(); const f = item.getAsFile(); if (f) addPendingImage(f); }
});

function clearChat() {
  if (running) stopConversation();
  if (isChatRecording) stopChatRecording();
  sessionId = null; currentBubble = null; currentBubbleText = ""; isGenerating = false;
  pendingImages = []; document.getElementById("imagePreviews").innerHTML = "";
  pendingAudios = []; document.getElementById("audioPreviews").innerHTML = "";
  chatEl.innerHTML = `
    <div class="empty-state" id="emptyState">
      <h2>Talk or type to Gemma 4</h2>
      <p>Type for rich chat — markdown, image upload, web search and object detection.
         Tap the mic for a hands-free spoken conversation.</p>
      <p class="hintline">Everything runs locally on this machine.</p>
    </div>`;
  document.getElementById("meterFill").style.width = "0%";
  document.getElementById("meterFill").classList.remove("warn", "danger");
  document.getElementById("meterText").textContent = "0 / 131K";
  fetch("/api/voice/reset", { method: "POST" }).catch(() => {});  // reset the spoken history too
  // Detach the old socket's handlers BEFORE closing so its onclose can't fire a
  // second reconnect (which would churn sessions).
  if (ws) { ws.onopen = ws.onmessage = ws.onerror = ws.onclose = null; try { ws.close(); } catch (e) {} }
  connect();
  updateControls();
}

// ----- model picker (the shared brain) -----
async function loadModels() {
  try {
    const data = await (await fetch("/models")).json();
    modelCapabilities = data.models;
    const sel = document.getElementById("modelSelect");
    sel.innerHTML = "";
    for (const [m, info] of Object.entries(data.models)) {
      const opt = document.createElement("option");
      opt.value = m; opt.textContent = info.label ? `${m} · ${info.label.split("·").pop().trim()}` : m;
      if (m === data.default) opt.selected = true;
      sel.appendChild(opt);
    }
    currentModel = data.default;
    updateChatMicState();
  } catch (e) { console.error("loadModels:", e); }
}
document.getElementById("modelSelect").addEventListener("change", (e) => {
  currentModel = e.target.value;
  updateChatMicState();
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: "set_model", model: currentModel }));
});

// ===========================================================================
//  VOICE — hands-free VAD orb  (lifted from real_time_voice/app.js, unchanged
//  in behaviour: arm mic, detect silence -> send clip -> play reply -> re-arm)
// ===========================================================================
let running = false;          // conversation active?
let turnBusy = false;         // a turn (fetch -> stream -> playback) is in flight
let state = "idle";           // idle | listening | thinking | speaking
let stream, audioCtx, analyser, dataArr;
let recorder, chunks = [];
let mimeType = "";
let vadTimer = null;
let scheduledSources = [];
let convAbort = null;

const FRAME_MS = 50, SILENCE_HANG_MS = 500, MIN_SPEECH_MS = 250, MAX_TURN_MS = 20000, CALIBRATE_MS = 400;
let noiseFloor = 0.01, speechThresh = 0.02;

function sensitivity() { const v = parseInt(sensEl.value, 10); return 3.2 - (v - 1) * 0.22; }

function setState(s) {
  state = s;
  orb.className = "orb " + s;
  orb.style.setProperty("--amp", "0");
  orb.setAttribute("aria-pressed", running ? "true" : "false");
  orb.setAttribute("aria-label",
    s === "listening" ? "Listening, tap to stop" : s === "thinking" ? "Thinking" :
    s === "speaking"  ? "Speaking" : running ? "Stop conversation" : "Start voice conversation");
  orbLabel.textContent = s === "listening" ? "Listening…" : s === "thinking" ? "Thinking…" :
    s === "speaking" ? "Speaking…" : running ? "Stop" : "Start";
  voiceHint.textContent =
    s === "listening" ? "Listening… talk, or tap the mic to stop." :
    s === "thinking"  ? "Thinking…" :
    s === "speaking"  ? "Speaking…" : "";
}

function beginTurn() { turnBusy = true; sendBtn.disabled = true; textarea.disabled = true; }
function endTurn() { turnBusy = false; sendBtn.disabled = isGenerating; textarea.disabled = false; }

function pickMime() {
  const opts = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4", "audio/ogg"];
  for (const o of opts) if (MediaRecorder.isTypeSupported(o)) return o;
  return "";
}
function ensureAudioCtx() {
  if (!audioCtx || audioCtx.state === "closed") audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  return audioCtx;
}
function unlockAudio() {
  ensureAudioCtx();
  if (audioCtx.state === "suspended") audioCtx.resume();
  try { const b = audioCtx.createBuffer(1, 1, 22050); const s = audioCtx.createBufferSource(); s.buffer = b; s.connect(audioCtx.destination); s.start(0); } catch (e) {}
}

async function startConversation() {
  running = true;
  stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true } });
  ensureAudioCtx();
  const src = audioCtx.createMediaStreamSource(stream);
  analyser = audioCtx.createAnalyser(); analyser.fftSize = 1024;
  dataArr = new Float32Array(analyser.fftSize);
  src.connect(analyser);
  mimeType = pickMime();
  armListening();
}

function stopConversation() {
  running = false;
  clearInterval(vadTimer);
  if (recorder && recorder.state !== "inactive") { recorder.onstop = null; recorder.stop(); }
  scheduledSources.forEach(s => { try { s.stop(); } catch (e) {} });
  scheduledSources = [];
  if (convAbort) { try { convAbort.abort(); } catch (e) {} convAbort = null; }
  endTurn();
  if (stream) stream.getTracks().forEach(t => t.stop());
  if (audioCtx) { try { audioCtx.close(); } catch (e) {} }
  setState("idle");
  voiceHint.textContent = "";
}

function rms() {
  analyser.getFloatTimeDomainData(dataArr);
  let s = 0; for (let i = 0; i < dataArr.length; i++) s += dataArr[i] * dataArr[i];
  return Math.sqrt(s / dataArr.length);
}

function armListening() {
  if (!running) return;
  setState("listening");
  chunks = [];
  recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
  recorder.ondataavailable = e => { if (e.data.size) chunks.push(e.data); };
  recorder.start(100);

  let calibrating = true, calStart = performance.now(), calSum = 0, calN = 0;
  let speechStart = null, lastVoice = null, turnStart = performance.now();

  clearInterval(vadTimer);
  vadTimer = setInterval(() => {
    if (!running || state !== "listening") return;
    const level = rms(), now = performance.now();
    if (calibrating) {
      calSum += level; calN++;
      if (now - calStart >= CALIBRATE_MS) {
        noiseFloor = Math.max(0.006, calSum / Math.max(1, calN));
        speechThresh = noiseFloor * sensitivity() + 0.004;
        calibrating = false;
      }
      return;
    }
    const voiced = level > speechThresh;
    orb.style.setProperty("--amp", Math.min(1, level * 9).toFixed(3));
    if (voiced) { if (speechStart === null) speechStart = now; lastVoice = now; }
    const spokeEnough = speechStart !== null && (lastVoice - speechStart) > MIN_SPEECH_MS;
    const wentQuiet = lastVoice !== null && (now - lastVoice) > SILENCE_HANG_MS;
    if (spokeEnough && wentQuiet) { finalizeTurn(); return; }
    if (now - turnStart > MAX_TURN_MS && speechStart !== null) { finalizeTurn(); return; }
  }, FRAME_MS);
}

function finalizeTurn() {
  clearInterval(vadTimer);
  setState("thinking");
  recorder.onstop = sendClip;
  if (recorder.state !== "inactive") recorder.stop();
}

async function sendClip() {
  const blob = new Blob(chunks, { type: mimeType || "audio/webm" });
  if (blob.size < 1200) { armListening(); return; }
  const fd = new FormData();
  const ext = mimeType.includes("mp4") ? "mp4" : (mimeType.includes("ogg") ? "ogg" : "webm");
  fd.append("audio", blob, "turn." + ext);
  setState("thinking"); beginTurn();
  convAbort = new AbortController();
  let resp;
  try {
    resp = await fetch("/api/converse_stream", { method: "POST", body: fd, signal: convAbort.signal });
  } catch (e) {
    endTurn(); if (!running) return;
    addMessage("assistant", "⚠️ Could not reach the local server.", null, { voice: true, plain: true });
    setState("error"); return;
  }
  await playStream(resp);
}

async function playStream(resp) {
  try { await audioCtx.resume(); } catch (e) {}
  const STARTUP_BUFFER_S = 0.4;
  const pending = [];
  let bufferedSec = 0, started = false, playhead = 0, lastEnd = 0, sawAudio = false, aiBubble = null;
  scheduledSources = [];

  function schedule(int16, sr) {
    if (playhead === 0) playhead = audioCtx.currentTime + 0.06;
    const n = int16.length, f32 = new Float32Array(n);
    for (let i = 0; i < n; i++) f32[i] = int16[i] / 32768;
    const ab = audioCtx.createBuffer(1, n, sr); ab.copyToChannel(f32, 0);
    const src = audioCtx.createBufferSource(); src.buffer = ab; src.connect(audioCtx.destination);
    const at = Math.max(playhead, audioCtx.currentTime + 0.02);
    src.start(at); scheduledSources.push(src);
    playhead = at + ab.duration; lastEnd = playhead;
  }
  function startIfReady(force) {
    if (!started && (force || bufferedSec >= STARTUP_BUFFER_S)) { started = true; setState("speaking"); }
    if (started) { while (pending.length) { const a = pending.shift(); schedule(a.int16, a.sr); } }
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
        if (ev.type === "transcript") { if (ev.text) addMessage("user", ev.text, null, { plain: true }); }
        else if (ev.type === "reply_text") { aiBubble = addMessage("assistant", ev.text || "…", null, { voice: true, plain: true }); }
        else if (ev.type === "audio") {
          const int16 = b64ToInt16(ev.pcm), sr = ev.sr || 24000;
          sawAudio = true; pending.push({ int16, sr });
          bufferedSec += int16.length / sr; startIfReady(false);
        } else if (ev.type === "error") {
          const msg = "⚠️ " + (ev.message || "Something went wrong.");
          if (aiBubble) aiBubble.querySelector(".bubble-content").textContent = msg;
          else addMessage("assistant", msg, null, { voice: true, plain: true });
        }
      }
    }
  } catch (e) {
    if (convAbort && convAbort.signal.aborted) { endTurn(); return; }
  }
  if (convAbort && convAbort.signal.aborted) { endTurn(); return; }
  startIfReady(true);
  const finish = () => { endTurn(); if (running) armListening(); else setState("idle"); };
  if (!sawAudio) { finish(); return; }
  const waitMs = Math.max(0, (lastEnd - audioCtx.currentTime) * 1000) + 150;
  setTimeout(finish, waitMs);
}

function b64ToInt16(b64) {
  const bin = atob(b64), u8 = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  return new Int16Array(u8.buffer, 0, u8.length >> 1);
}

// orb wiring
orb.addEventListener("click", async () => {
  if (isGenerating) return;        // a text turn is streaming
  unlockAudio();
  if (!running) {
    try { await startConversation(); }
    catch (e) { voiceHint.textContent = "Mic blocked — allow microphone access for this site."; setState("error"); }
  } else { stopConversation(); }
});
orb.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); orb.click(); } });

// ===========================================================================
//  Settings (voice engine / voice / personality / mic sensitivity)
// ===========================================================================
let defaultPrompt = "";
const engineSel = document.getElementById("engineSel");
const voiceSel  = document.getElementById("voiceSel");

function fillVoices(engine, selected) {
  const list = engine === "orpheus" ? ORPHEUS_VOICES : KOKORO_VOICES;
  voiceSel.innerHTML = "";
  for (const v of list) {
    const o = document.createElement("option"); o.value = v; o.textContent = v;
    if (v === selected) o.selected = true;
    voiceSel.appendChild(o);
  }
}

async function loadConfig() {
  try {
    const c = await (await fetch("/api/voice/config")).json();
    defaultPrompt = c.default_system_prompt || "";
    engineSel.value = c.engine || "kokoro";
    fillVoices(engineSel.value, c.voice);
    document.getElementById("sysprompt").value = c.system_prompt || "";
  } catch (e) {}
}

engineSel.addEventListener("change", () => fillVoices(engineSel.value, null));

async function applyConfig() {
  const st = document.getElementById("cfgStatus");
  st.textContent = "saving…";
  try {
    await fetch("/api/voice/config", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        engine: engineSel.value, voice: voiceSel.value,
        system_prompt: document.getElementById("sysprompt").value,
      }),
    });
    st.textContent = "✓ applied — takes effect next spoken turn";
    setTimeout(() => { st.textContent = ""; }, 2500);
  } catch (e) { st.textContent = "⚠️ failed"; }
}
document.getElementById("applyCfg").addEventListener("click", applyConfig);
document.getElementById("resetCfg").addEventListener("click", () => {
  document.getElementById("sysprompt").value = defaultPrompt; applyConfig();
});

// close the settings panel when tapping outside (nice on mobile)
document.addEventListener("click", (e) => {
  const s = document.getElementById("settings");
  if (s.open && !s.contains(e.target)) s.removeAttribute("open");
});

// ===========================================================================
//  Init
// ===========================================================================
document.getElementById("chatMicBtn")?.addEventListener("click", toggleChatRecording);

loadModels();
connect();
loadConfig();
setState("idle");
textarea.focus();
