// --- Config ---
const API_URL = window.API_URL || "http://127.0.0.1:8000/predict";

// --- DOM refs ---
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const brush = document.getElementById("brush");
const clearBtn = document.getElementById("clearBtn");
const undoBtn = document.getElementById("undoBtn");
const predictBtn = document.getElementById("predictBtn");
const fileInput = document.getElementById("fileInput");
const statusEl = document.getElementById("status");
const probsEl = document.getElementById("probs");
const predEl = document.getElementById("pred");
const loading = document.getElementById("loading");
const apiUrlEl = document.getElementById("apiUrl");
const themeToggle = document.getElementById("themeToggle");

// Basic page init
apiUrlEl.textContent = API_URL;
document.getElementById("year").textContent = new Date().getFullYear();

// --- Theme toggle (auto / manual) ---
const root = document.documentElement;
let manualTheme = localStorage.getItem("theme"); // "light" | "dark" | null
applyTheme(manualTheme || "auto");
themeToggle.addEventListener("click", () => {
  const next =
    manualTheme === "dark" ? "light" :
    manualTheme === "light" ? "auto" : "dark";
  manualTheme = next === "auto" ? null : next;
  if (manualTheme === null) localStorage.removeItem("theme");
  else localStorage.setItem("theme", manualTheme);
  applyTheme(next);
});
function applyTheme(mode) {
  if (mode === "auto") root.setAttribute("data-theme", "auto");
  else root.setAttribute("data-theme", mode);
}

// --- Helpers (status, spinner, prediction, bars) ---
function setStatus(msg) { statusEl.innerHTML = msg; }
function setPred(n) { predEl.textContent = typeof n === "number" ? String(n) : "–"; }
function showLoading(show) {
  loading.hidden = !show;
  loading.setAttribute("aria-busy", String(show));
}
function renderProbs(probs, bestIdx = -1) {
  probsEl.innerHTML = "";
  for (let i = 0; i < 10; i++) {
    const p = Number(probs?.[i] ?? 0);
    const row = document.createElement("div");
    row.className = "prob-row" + (i === bestIdx ? " best" : "");
    row.setAttribute("role", "listitem");
    const lab = document.createElement("div"); lab.textContent = i;
    const bar = document.createElement("div"); bar.className = "prob-bar";
    const pct = Math.max(0, Math.min(1, p)); bar.style.setProperty("--p", pct);
    const val = document.createElement("div"); val.textContent = (p * 100).toFixed(1) + "%";
    row.append(lab, bar, val); probsEl.append(row);
  }
}

// Safety: ensure spinner is OFF on initial load
showLoading(false);
setStatus('Ready. Draw a digit or upload an image, then click <strong>Predict</strong>.');
setPred(null);
renderProbs([]);

// =========================
// Drawing state (declare first)
// =========================
let drawing = false;            // are we currently drawing?
const strokeHistory = [];       // snapshots for undo

const DPR = Math.max(1, Math.min(2, window.devicePixelRatio || 1));
function resetCanvas() {
  const w = 280, h = 280;
  canvas.width = w * DPR;       // real pixel size
  canvas.height = h * DPR;
  canvas.style.width = w + "px";   // CSS size
  canvas.style.height = h + "px";

  // draw using CSS pixels (scale context)
  ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, w, h);     // black background
  ctx.strokeStyle = "#fff";     // white brush
  ctx.lineWidth = Number(brush.value || 18);
  ctx.lineCap = "round";
  ctx.lineJoin = "round";

  strokeHistory.length = 0;     // clear undo stack
  saveSnapshot();                // baseline snapshot
}

if (!ctx) {
  setStatus("⚠️ Your browser doesn't support Canvas 2D.");
} else {
  canvas.style.touchAction = "none"; // prevent page panning while drawing
  resetCanvas();
}

// =========================
// Drawing (with undo stack)
// =========================
function readImageDataFull() {
  // read pixels in real (device) coordinates
  ctx.save();
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  const img = ctx.getImageData(0, 0, canvas.width, canvas.height);
  ctx.restore();
  return img;
}
function writeImageDataFull(img) {
  // write pixels in real coordinates
  ctx.save();
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.putImageData(img, 0, 0);
  ctx.restore();
  // restore drawing transform
  ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
}
function saveSnapshot() {
  if (strokeHistory.length > 30) strokeHistory.shift(); // limit memory
  strokeHistory.push(readImageDataFull());
}
function undo() {
  if (strokeHistory.length <= 1) return; // keep baseline
  strokeHistory.pop();
  const img = strokeHistory[strokeHistory.length - 1];
  writeImageDataFull(img);
}

function pos(e) {
  // get cursor/touch position relative to canvas (CSS pixels)
  const r = canvas.getBoundingClientRect();
  const clientX = e.touches ? e.touches[0].clientX : e.clientX;
  const clientY = e.touches ? e.touches[0].clientY : e.clientY;
  return { x: clientX - r.left, y: clientY - r.top };
}
function startDraw(e) {
  e.preventDefault();
  drawing = true;
  const { x, y } = pos(e);
  ctx.beginPath();
  ctx.moveTo(x, y);
}
function moveDraw(e) {
  if (!drawing) return;
  e.preventDefault();
  const { x, y } = pos(e);
  ctx.lineTo(x, y);
  ctx.stroke();
}
function endDraw() {
  if (!drawing) return;
  drawing = false;
  ctx.closePath();
  saveSnapshot();
}

// Mouse events
canvas.addEventListener("mousedown", startDraw);
canvas.addEventListener("mousemove", moveDraw);
canvas.addEventListener("mouseup", endDraw);
canvas.addEventListener("mouseleave", endDraw);

// Touch events
canvas.addEventListener("touchstart", startDraw, { passive: false });
canvas.addEventListener("touchmove", moveDraw, { passive: false });
canvas.addEventListener("touchend", endDraw);

// Brush size slider
brush.addEventListener("input", () => (ctx.lineWidth = Number(brush.value)));

// --- Controls ---
clearBtn.addEventListener("click", () => {
  resetCanvas();
  setStatus("Canvas cleared.");
  setPred(null);
  renderProbs([]);
});
undoBtn.addEventListener("click", () => {
  undo();
  setStatus("Undid last stroke.");
});
document.addEventListener("keydown", (e) => {
  const k = e.key?.toLowerCase();
  if (k === "c") clearBtn.click();
  if (k === "z") undoBtn.click();
});

// --- File upload -> predict ---
fileInput.addEventListener("change", async (e) => {
  const file = e.target.files?.[0];
  if (!file) return;
  await sendForPrediction(file);
  fileInput.value = "";
});

// --- Predict button (canvas -> blob -> API) ---
predictBtn.addEventListener("click", async () => {
  const blob = await canvasToBlob(canvas);
  await sendForPrediction(blob, "digit.png");
});

// --- Networking ---
function canvasToBlob(cnv) {
  return new Promise((resolve) => cnv.toBlob(resolve, "image/png"));
}

async function sendForPrediction(fileOrBlob, filename = "image.png") {
  try {
    showLoading(true);
    setStatus("Predicting…");

    const formData = new FormData();
    formData.append("file", fileOrBlob, filename);

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 10000);

    const res = await fetch(API_URL, {
      method: "POST",
      body: formData,
      signal: controller.signal,
    });
    clearTimeout(timeoutId);

    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    const pred = Number.isFinite(data.prediction)
      ? data.prediction
      : Number.isFinite(data.label)
      ? data.label
      : null;

    const probs = Array.isArray(data.probabilities)
      ? data.probabilities
      : Array.isArray(data.probs)
      ? data.probs
      : [];

    setPred(pred);
    renderProbs(probs, pred ?? -1);
    setStatus(pred !== null ? `Prediction: ${pred}` : "Prediction unavailable.");
  } catch (err) {
    console.error(err);
    const aborted = err && err.name === "AbortError";
    setStatus(
      aborted
        ? "⚠️ Request timed out. Is the API running at the configured URL?"
        : "⚠️ Error: could not reach API. Is it running and CORS-allowed?"
    );
    setPred(null);
    renderProbs([]);
  } finally {
    showLoading(false);
  }
}