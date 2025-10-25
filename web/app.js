// ---- Config ----
const API_URL = "http://127.0.0.1:8000/predict";

// ---- Elements ----
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const clearBtn = document.getElementById("clearBtn");
const predictBtn = document.getElementById("predictBtn");
const fileInput = document.getElementById("fileInput");
const statusEl = document.getElementById("status");
const probsEl = document.getElementById("probs");
document.getElementById("apiUrl").textContent = API_URL;

// ---- Canvas drawing setup ----
const pen = {
  drawing: false,
  lastX: 0,
  lastY: 0,
  lineWidth: 18,
  strokeStyle: "#ffffff"
};

function resetCanvas() {
  ctx.fillStyle = "#000000";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.strokeStyle = pen.strokeStyle;
  ctx.lineWidth = pen.lineWidth;
}
resetCanvas();

function drawLine(x, y) {
  ctx.lineTo(x, y);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(x, y);
}

function pointerPos(e) {
  const r = canvas.getBoundingClientRect();
  const x = (e.touches ? e.touches[0].clientX : e.clientX) - r.left;
  const y = (e.touches ? e.touches[0].clientY : e.clientY) - r.top;
  return { x, y };
}

function startDraw(e) {
  e.preventDefault();
  pen.drawing = true;
  const { x, y } = pointerPos(e);
  pen.lastX = x;
  pen.lastY = y;
  ctx.beginPath();
  ctx.moveTo(x, y);
}
function moveDraw(e) {
  if (!pen.drawing) return;
  e.preventDefault();
  const { x, y } = pointerPos(e);
  drawLine(x, y);
}
function endDraw(e) {
  if (!pen.drawing) return;
  e.preventDefault();
  pen.drawing = false;
  ctx.beginPath();
}

canvas.addEventListener("mousedown", startDraw);
canvas.addEventListener("mousemove", moveDraw);
canvas.addEventListener("mouseup", endDraw);
canvas.addEventListener("mouseleave", endDraw);
canvas.addEventListener("touchstart", startDraw, { passive: false });
canvas.addEventListener("touchmove", moveDraw, { passive: false });
canvas.addEventListener("touchend", endDraw);

// ---- UI actions ----
clearBtn.addEventListener("click", () => {
  resetCanvas();
  statusEl.textContent = "Canvas cleared.";
  probsEl.innerHTML = "";
});

fileInput.addEventListener("change", async (e) => {
  const file = e.target.files?.[0];
  if (!file) return;
  await sendForPrediction(file);
  fileInput.value = "";
});

predictBtn.addEventListener("click", async () => {
  const blob = await canvasToBlob(canvas);
  await sendForPrediction(blob, "digit.png");
});

// ---- Helpers ----
function canvasToBlob(cnv) {
  return new Promise((resolve) => cnv.toBlob(resolve, "image/png"));
}

function renderProbs(probs, pred) {
  probsEl.innerHTML = "";
  probs.forEach((p, i) => {
    const d = document.createElement("div");
    d.className = "prob" + (i === pred ? " best" : "");
    d.textContent = `${i}: ${p.toFixed(3)}`;
    probsEl.appendChild(d);
  });
}

async function sendForPrediction(fileOrBlob, filename = "image.png") {
  try {
    statusEl.textContent = "Predicting…";
    const formData = new FormData();
    formData.append("file", fileOrBlob, filename);

    const res = await fetch(API_URL, { method: "POST", body: formData });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    const pred = data.prediction ?? data.label ?? 0;
    const probs = data.probabilities ?? data.probs ?? [];
    statusEl.textContent = `Prediction: ${pred}`;
    if (Array.isArray(probs) && probs.length === 10) renderProbs(probs, pred);
    else probsEl.innerHTML = "";
  } catch (err) {
    console.error(err);
    statusEl.textContent = "Error: could not reach API (is it running?)";
    probsEl.innerHTML = "";
  }
}
