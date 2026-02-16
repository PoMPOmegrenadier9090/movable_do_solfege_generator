// ─── DOM elements ───
const form = document.getElementById("upload-form");
const clientInput = document.getElementById("client-id");
const fileInput = document.getElementById("audio-file");

const jobIdEl = document.getElementById("job-id");
const jobStatusEl = document.getElementById("job-status");
const jobStepEl = document.getElementById("job-step");
const progressBarEl = document.getElementById("progress-bar");
const estimatedKeyEl = document.getElementById("estimated-key");
const mixAudioEl = document.getElementById("mix-audio");
const midiLinkEl = document.getElementById("midi-link");
const jsonLinkEl = document.getElementById("json-link");

const container = document.getElementById("piano-roll-container");
const canvas = document.getElementById("piano-roll");
const ctx = canvas.getContext("2d");
const labelCanvas = document.getElementById("piano-roll-labels");
const labelCtx = labelCanvas.getContext("2d");

// ─── Piano roll constants ───
const ROW_HEIGHT = 22; // 各ピッチ行の高さ
const LABEL_WIDTH = 90; // 左端のピッチラベル幅（絶対音名+ソルフェージュ）
const TOP_PADDING = 10;
const VISIBLE_DURATION = 45; // ビューポートに表示する秒数
const PLAYHEAD_COLOR = "rgba(255, 60, 60, 0.9)";
const NOTE_COLORS = [
  "#6366f1", "#818cf8", "#a78bfa", "#c084fc",
  "#e879f9", "#f472b6", "#fb7185", "#f87171",
  "#fb923c", "#fbbf24", "#a3e635", "#34d399",
];
const NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
const SOLFEGE_NAMES = ["ド", "ド#", "レ", "レ#", "ミ", "ファ", "ファ#", "ソ", "ソ#", "ラ", "ラ#", "シ"];
// config.py の歌唱音域: LOWEST_PITCH=15 (MIDI 36=C2), HIGHEST_PITCH=67 (MIDI 88=E6)
const MIDI_OFFSET = 21;
const FIXED_MIN_PITCH = 15 + MIDI_OFFSET; // MIDI 36 = C2
const FIXED_MAX_PITCH = 67 + MIDI_OFFSET; // MIDI 88 = E6

// ─── State ───
let sortedNotes = []; // ノートを start 時間でソート済み
let keySequence = []; // [{grid_time, key}, ...] from API
let minPitch = FIXED_MIN_PITCH;
let maxPitch = FIXED_MAX_PITCH;
let totalDuration = 0;
let canvasHeight = 0;
let animFrameId = null;
let viewCenterTime = 0; // ビューポートの中央時間
let currentKeyLabel = ""; // 現在のプレイヘッド位置のキー
let labelCacheCanvas = null; // ラベル背景キャッシュ
let pianoRollReady = false; // ノートデータがロード済みか

// ─── Utility ───
function setStatus(status, step, progress = 0) {
  jobStatusEl.textContent = `Status: ${status}`;
  jobStepEl.textContent = `Step: ${step || "-"}`;
  progressBarEl.style.width = `${Math.max(0, Math.min(100, progress * 100))}%`;
}

function pitchToNoteName(pitch) {
  const octave = Math.floor(pitch / 12) - 1;
  return `${NOTE_NAMES[pitch % 12]}${octave}`;
}

function pitchToY(pitch) {
  return TOP_PADDING + (maxPitch - pitch) * ROW_HEIGHT;
}

// ─── Viewport time ↔ pixel mapping ───
function getPixelsPerSecond() {
  return canvas.width / VISIBLE_DURATION;
}

function timeToX(time) {
  const pps = getPixelsPerSecond();
  return (time - (viewCenterTime - VISIBLE_DURATION / 2)) * pps;
}

function xToTime(x) {
  const pps = getPixelsPerSecond();
  return (x / pps) + (viewCenterTime - VISIBLE_DURATION / 2);
}

// ─── Key / solfege helpers ───
function keyLabelToSolfegeRoot(keyLabel) {
  if (!keyLabel) return 0;
  const parts = keyLabel.split(" ");
  const root = parts[0];
  const idx = NOTE_NAMES.indexOf(root);
  const rootPc = idx >= 0 ? idx : 0;
  if (parts[1] === "Minor") {
    return (rootPc + 3) % 12;
  }
  return rootPc;
}

function getKeyAtTime(time) {
  if (keySequence.length === 0) return "";
  let key = keySequence[0].key;
  for (const ks of keySequence) {
    if (ks.grid_time <= time) {
      key = ks.key;
    } else {
      break;
    }
  }
  return key;
}

function pitchToSolfege(pitch, rootPc) {
  const interval = ((pitch % 12) - rootPc + 12) % 12;
  return SOLFEGE_NAMES[interval];
}

// ─── Binary search: visible notes in [startTime, endTime] ───
// sortedNotes is sorted by `start`. We find all notes where:
//   note.end > viewStart AND note.start < viewEnd
function findVisibleNotes(viewStart, viewEnd) {
  if (sortedNotes.length === 0) return [];

  // Binary search: find first note where start < viewEnd
  // (notes that start after viewEnd are definitely not visible)
  let lo = 0, hi = sortedNotes.length;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (sortedNotes[mid].start < viewStart - 10) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }

  // Collect visible notes from lo onwards
  const visible = [];
  for (let i = Math.max(0, lo - 1); i < sortedNotes.length; i++) {
    const note = sortedNotes[i];
    if (note.start >= viewEnd) break; // past the view
    if (note.end > viewStart) {
      visible.push(note);
    }
  }
  return visible;
}

// ─── Label canvas: ラベル背景をキャッシュ ───
function renderLabelCache(height) {
  labelCacheCanvas = document.createElement("canvas");
  labelCacheCanvas.width = LABEL_WIDTH;
  labelCacheCanvas.height = height;
  const lCtx = labelCacheCanvas.getContext("2d");

  lCtx.fillStyle = "#1a1a2e";
  lCtx.fillRect(0, 0, LABEL_WIDTH, height);

  for (let p = minPitch; p <= maxPitch; p++) {
    const y = pitchToY(p);
    const isC = p % 12 === 0;

    if (isC) {
      lCtx.strokeStyle = "rgba(255,255,255,0.15)";
      lCtx.lineWidth = 1;
      lCtx.beginPath();
      lCtx.moveTo(0, y + ROW_HEIGHT);
      lCtx.lineTo(LABEL_WIDTH, y + ROW_HEIGHT);
      lCtx.stroke();
    }

    const pc = p % 12;
    if ([1, 3, 6, 8, 10].includes(pc)) {
      lCtx.fillStyle = "rgba(0,0,0,0.15)";
      lCtx.fillRect(0, y, LABEL_WIDTH, ROW_HEIGHT);
    }
  }

  lCtx.strokeStyle = "rgba(255,255,255,0.1)";
  lCtx.lineWidth = 1;
  lCtx.beginPath();
  lCtx.moveTo(LABEL_WIDTH - 0.5, 0);
  lCtx.lineTo(LABEL_WIDTH - 0.5, height);
  lCtx.stroke();
}

function updateLabels(rootPc) {
  if (!labelCacheCanvas) return;

  labelCanvas.width = LABEL_WIDTH;
  labelCanvas.height = canvasHeight;

  labelCtx.drawImage(labelCacheCanvas, 0, 0);

  for (let p = minPitch; p <= maxPitch; p++) {
    const y = pitchToY(p);
    const isC = p % 12 === 0;
    const noteName = pitchToNoteName(p);
    const solfege = pitchToSolfege(p, rootPc);

    // 絶対音名
    labelCtx.fillStyle = isC ? "#e2e8f0" : "rgba(255,255,255,0.4)";
    labelCtx.font = isC ? "bold 11px monospace" : "11px monospace";
    labelCtx.textBaseline = "middle";
    labelCtx.fillText(noteName, 4, y + ROW_HEIGHT / 2);

    // 相対音名 (ソルフェージュ)
    labelCtx.fillStyle = isC ? "#a78bfa" : "rgba(167,139,250,0.5)";
    labelCtx.font = "bold 10px sans-serif";
    labelCtx.fillText(solfege, 46, y + ROW_HEIGHT / 2);
  }
}

// ─── Main render: draw visible notes + playhead ───
function renderFrame() {
  if (!pianoRollReady) return;

  const currentTime = mixAudioEl.currentTime;
  viewCenterTime = currentTime;

  const viewStart = viewCenterTime - VISIBLE_DURATION / 2;
  const viewEnd = viewCenterTime + VISIBLE_DURATION / 2;
  const pps = getPixelsPerSecond();
  const w = canvas.width;
  const h = canvas.height;

  // === Clear + background ===
  ctx.fillStyle = "#1a1a2e";
  ctx.fillRect(0, 0, w, h);

  // === Grid lines + black key shading ===
  for (let p = minPitch; p <= maxPitch; p++) {
    const y = pitchToY(p);
    const isC = p % 12 === 0;

    ctx.strokeStyle = isC ? "rgba(255,255,255,0.15)" : "rgba(255,255,255,0.05)";
    ctx.lineWidth = isC ? 1 : 0.5;
    ctx.beginPath();
    ctx.moveTo(0, y + ROW_HEIGHT);
    ctx.lineTo(w, y + ROW_HEIGHT);
    ctx.stroke();

    const pc = p % 12;
    if ([1, 3, 6, 8, 10].includes(pc)) {
      ctx.fillStyle = "rgba(0,0,0,0.15)";
      ctx.fillRect(0, y, w, ROW_HEIGHT);
    }
  }

  // === Time grid (vertical lines every second) ===
  ctx.font = "10px monospace";
  ctx.textBaseline = "top";
  const startSec = Math.floor(Math.max(0, viewStart));
  const endSec = Math.ceil(Math.min(totalDuration, viewEnd));
  for (let t = startSec; t <= endSec; t++) {
    const x = timeToX(t);
    if (x < 0 || x > w) continue;
    const isMajor = t % 5 === 0;
    ctx.strokeStyle = isMajor ? "rgba(255,255,255,0.12)" : "rgba(255,255,255,0.04)";
    ctx.lineWidth = isMajor ? 1 : 0.5;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, h);
    ctx.stroke();
    if (isMajor) {
      const mins = Math.floor(t / 60);
      const secs = t % 60;
      ctx.fillStyle = "rgba(255,255,255,0.2)";
      ctx.fillText(`${mins}:${secs.toString().padStart(2, "0")}`, x + 2, 2);
    }
  }

  // === Visible notes ===
  const visible = findVisibleNotes(viewStart, viewEnd);

  // 重複排除: 可視範囲の直前のノートのピッチを初期値にする
  let prevPitch = -1;
  if (visible.length > 0) {
    const firstIdx = sortedNotes.indexOf(visible[0]);
    if (firstIdx > 0) {
      prevPitch = sortedNotes[firstIdx - 1].pitch;
    }
  }

  for (const note of visible) {
    const x = timeToX(note.start);
    const noteW = Math.max((note.end - note.start) * pps, 3);
    const y = pitchToY(note.pitch);
    const colorIdx = note.pitch % 12;

    // Note body
    ctx.fillStyle = NOTE_COLORS[colorIdx];
    ctx.beginPath();
    ctx.roundRect(x, y + 1, noteW, ROW_HEIGHT - 2, 3);
    ctx.fill();

    // Solfege label above (同じピッチが連続する場合は最初の音符のみ)
    if (note.solfege && note.pitch !== prevPitch) {
      ctx.fillStyle = "#e2e8f0";
      ctx.font = "bold 11px sans-serif";
      ctx.textBaseline = "bottom";
      ctx.fillText(note.solfege, x, y - 1);
    }

    prevPitch = note.pitch;
  }

  // === Playhead (always at center) ===
  const playheadX = timeToX(currentTime);

  // Glow
  const grad = ctx.createLinearGradient(playheadX - 8, 0, playheadX + 8, 0);
  grad.addColorStop(0, "rgba(255, 60, 60, 0)");
  grad.addColorStop(0.5, "rgba(255, 60, 60, 0.15)");
  grad.addColorStop(1, "rgba(255, 60, 60, 0)");
  ctx.fillStyle = grad;
  ctx.fillRect(playheadX - 8, 0, 16, h);

  // Line
  ctx.strokeStyle = PLAYHEAD_COLOR;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(playheadX, 0);
  ctx.lineTo(playheadX, h);
  ctx.stroke();

  // Time label
  const mins = Math.floor(currentTime / 60);
  const secs = Math.floor(currentTime % 60);
  const timeStr = `${mins}:${secs.toString().padStart(2, "0")}`;
  ctx.fillStyle = "rgba(255, 60, 60, 0.9)";
  ctx.font = "bold 11px sans-serif";
  ctx.textBaseline = "top";
  ctx.fillText(timeStr, playheadX + 4, 4);

  animFrameId = requestAnimationFrame(renderFrame);
}

function startAnimation() {
  if (animFrameId) cancelAnimationFrame(animFrameId);
  animFrameId = requestAnimationFrame(renderFrame);
}

function stopAnimation() {
  if (animFrameId) {
    cancelAnimationFrame(animFrameId);
    animFrameId = null;
  }
  // Final frame
  if (pianoRollReady) {
    renderFrame();
  }
}

// ─── Setup piano roll from result data ───
function setupPianoRoll(notes, keySeq) {
  const rawNotes = notes || [];
  keySequence = keySeq || [];

  if (rawNotes.length === 0) {
    canvas.width = container.clientWidth;
    canvas.height = 400;
    ctx.fillStyle = "#1a1a2e";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#94a3b8";
    ctx.font = "16px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("No notes detected", canvas.width / 2, canvas.height / 2);
    pianoRollReady = false;
    return;
  }

  // Sort notes by start time, then by pitch (for deduplication)
  sortedNotes = [...rawNotes].sort((a, b) => a.start - b.start || a.pitch - b.pitch);
  totalDuration = Math.max(...rawNotes.map((n) => n.end));

  // Fixed pitch range
  minPitch = FIXED_MIN_PITCH;
  maxPitch = FIXED_MAX_PITCH;

  // Canvas sizing: width = container width (viewport), height = pitch range
  const pitchRange = maxPitch - minPitch + 1;
  canvasHeight = TOP_PADDING * 2 + pitchRange * ROW_HEIGHT;
  canvas.width = container.clientWidth;
  canvas.height = canvasHeight;

  // Render label cache
  renderLabelCache(canvasHeight);
  const initialKey = keySequence.length > 0 ? keySequence[0].key : "C Major";
  currentKeyLabel = initialKey;
  updateLabels(keyLabelToSolfegeRoot(initialKey));

  pianoRollReady = true;
  viewCenterTime = 0;

  // Initial draw
  renderFrame();
}

// ─── Handle container resize ───
const resizeObserver = new ResizeObserver(() => {
  if (!pianoRollReady) return;
  canvas.width = container.clientWidth;
  canvas.height = canvasHeight;
  if (mixAudioEl.paused) {
    renderFrame();
  }
});
resizeObserver.observe(container);

// ─── Audio event listeners ───
mixAudioEl.addEventListener("play", () => {
  startAnimation();
});

mixAudioEl.addEventListener("pause", () => {
  stopAnimation();
});

mixAudioEl.addEventListener("ended", () => {
  stopAnimation();
});

mixAudioEl.addEventListener("seeked", () => {
  if (mixAudioEl.paused && pianoRollReady) {
    viewCenterTime = mixAudioEl.currentTime;
    renderFrame();
  }
});

// ─── Click on piano roll to seek ───
canvas.addEventListener("click", (e) => {
  if (!mixAudioEl.src || totalDuration === 0) return;
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const clickX = (e.clientX - rect.left) * scaleX;
  const time = xToTime(clickX);
  if (time >= 0 && time <= totalDuration) {
    mixAudioEl.currentTime = time;
    if (mixAudioEl.paused) {
      viewCenterTime = time;
      renderFrame();
    }
  }
});

// ─── Fetch helper ───
async function fetchJson(url, options) {
  const response = await fetch(url, options);
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || `Request failed: ${response.status}`);
  }
  return payload;
}

// ─── Form submit ───
form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!fileInput.files || fileInput.files.length === 0) {
    alert("Audio file is required");
    return;
  }

  const payload = new FormData();
  payload.append("client_id", clientInput.value || "default_client");
  payload.append("file", fileInput.files[0]);

  try {
    setStatus("running", "processing", 0.1);
    const result = await fetchJson("/api/jobs", { method: "POST", body: payload });
    jobIdEl.textContent = `Job: ${result.job_id}`;
    estimatedKeyEl.textContent = `Estimated Key: ${result.estimated_global_key}`;
    mixAudioEl.src = result.media.mix_audio;
    midiLinkEl.href = result.media.midi;
    jsonLinkEl.href = result.media.solfege_json;
    setupPianoRoll(result.notes || [], result.key_sequence || []);
    setStatus("completed", "done", 1.0);
  } catch (err) {
    alert(err.message);
    setStatus("failed", "error", 1.0);
  }
});
