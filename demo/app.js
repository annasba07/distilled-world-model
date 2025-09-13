const API_URL = 'http://localhost:8000';

// Elements
const canvas = document.getElementById('frameCanvas');
const ctx = canvas.getContext('2d');
const promptEl = document.getElementById('prompt');
const btnGenerate = document.getElementById('btnGenerate');
const btnReset = document.getElementById('btnReset');
const btnDownload = document.getElementById('btnDownload');
const btnRecord = document.getElementById('btnRecord');
const btnCopy = document.getElementById('btnCopy');
const overlay = document.getElementById('overlay');
const healthDot = document.getElementById('healthDot');
const healthText = document.getElementById('healthText');
const fpsValue = document.getElementById('fpsValue');
const latencyValue = document.getElementById('latencyValue');
const framesValue = document.getElementById('framesValue');
const sessionIdEl = document.getElementById('sessionId');
const fpsSpark = document.getElementById('fpsSpark');
const timeline = document.getElementById('timeline');
const themeToggle = document.getElementById('themeToggle');
const toast = document.getElementById('toast');
const recBadge = document.getElementById('recBadge');

// State
let ws = null;
let sessionId = null;
let isBusy = false;
let fpsHistory = [];
let frameHistory = [];
let pressedKeys = new Set();
let lastActionAt = 0;
const ACTION_COOLDOWN_MS = 40; // throttle to avoid spam
let latestMetrics = null;
let recording = false;
let mediaRecorder = null;
let recordedChunks = [];

const actionMap = {
  'w': 0, 'a': 1, 's': 2, 'd': 3,
  ' ': 4,
};

function showToast(msg) {
  toast.textContent = msg;
  toast.classList.remove('hidden');
  setTimeout(() => toast.classList.add('hidden'), 2000);
}

async function healthCheck() {
  try {
    const res = await fetch(`${API_URL}/health`);
    const data = await res.json();
    const ok = data.status === 'healthy' && data.engine_loaded;
    healthDot.style.background = ok ? 'var(--success)' : 'var(--warn)';
    healthText.textContent = ok ? `Ready • CUDA: ${data.cuda_available}` : 'Initializing…';
  } catch {
    healthDot.style.background = 'var(--danger)';
    healthText.textContent = 'Server down';
  }
}
setInterval(healthCheck, 5000);
healthCheck();

function setBusy(v) {
  isBusy = v;
  overlay.classList.toggle('hidden', !v);
}

function drawSparkline(canvas, values) {
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0,0,w,h);
  if (!values.length) return;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const pad = 4;
  ctx.strokeStyle = '#7c8cf8';
  ctx.lineWidth = 2;
  ctx.beginPath();
  values.forEach((v, i) => {
    const x = pad + (i / (values.length - 1)) * (w - pad * 2);
    const y = h - pad - ((v - min) / Math.max(1e-6, (max - min))) * (h - pad * 2);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();
}

function displayFrame(base64Frame) {
  const img = new Image();
  img.onload = () => {
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    drawMetricsOverlay();
  };
  img.src = 'data:image/png;base64,' + base64Frame;
  // push to history
  frameHistory.push(base64Frame);
  if (frameHistory.length > 40) frameHistory.shift();
  renderTimeline();
}

function drawMetricsOverlay() {
  if (!latestMetrics) return;
  const pad = 10;
  ctx.save();
  ctx.font = '12px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace';
  ctx.fillStyle = 'rgba(0,0,0,0.5)';
  ctx.fillRect(pad - 4, pad - 12, 120, 40);
  ctx.fillStyle = '#fff';
  const fps = latestMetrics.fps ? latestMetrics.fps.toFixed(1) : '-';
  const ms = latestMetrics.inference_time ? (latestMetrics.inference_time * 1000).toFixed(1) : '-';
  ctx.fillText(`FPS: ${fps}`, pad, pad);
  ctx.fillText(`ms: ${ms}`, pad, pad + 14);
  ctx.restore();
}

function renderTimeline() {
  timeline.innerHTML = '';
  frameHistory.forEach((b64, idx) => {
    const div = document.createElement('div');
    div.className = 'thumb';
    const img = new Image();
    img.src = 'data:image/png;base64,' + b64;
    img.alt = `Frame ${idx+1}`;
    div.appendChild(img);
    div.onclick = () => displayFrame(b64);
    timeline.appendChild(div);
  });
}

async function createSession() {
  try {
    setBusy(true);
    const res = await fetch(`${API_URL}/session/create`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt: promptEl.value || null })
    });
    const data = await res.json();
    sessionId = data.session_id;
    sessionIdEl.textContent = sessionId.slice(0,8) + '…';
    displayFrame(data.initial_frame);
    connectWebSocket();
    showToast('Session created');
  } catch (e) {
    console.error(e); showToast('Failed to create session');
  } finally { setBusy(false); }
}

function connectWebSocket() {
  if (ws) ws.close();
  ws = new WebSocket(`ws://localhost:8000/ws/${sessionId}`);
  ws.onopen = () => showToast('Connected');
  ws.onmessage = (event) => {
    const message = JSON.parse(event.data);
    if (message.type === 'frame') {
      displayFrame(message.data);
      if (message.metrics) {
        latestMetrics = message.metrics;
        const fps = latestMetrics.fps || 0;
        const ms = (latestMetrics.inference_time || 0) * 1000;
        fpsValue.textContent = fps.toFixed(1);
        latencyValue.textContent = `${ms.toFixed(1)} ms`;
        framesValue.textContent = message.frame_number;
        fpsHistory.push(fps);
        if (fpsHistory.length > 50) fpsHistory.shift();
        drawSparkline(fpsSpark, fpsHistory);
      }
    } else if (message.type === 'reset') {
      displayFrame(message.data);
      framesValue.textContent = '1';
    }
  };
  ws.onclose = () => showToast('Disconnected');
  ws.onerror = () => showToast('WebSocket error');
}

function sendAction(action) {
  const now = performance.now();
  if (now - lastActionAt < ACTION_COOLDOWN_MS) return;
  lastActionAt = now;
  if (ws && ws.readyState === WebSocket.OPEN && !isBusy) {
    ws.send(JSON.stringify({ type: 'action', action }));
  }
}

// Handlers
btnGenerate.addEventListener('click', createSession);
btnReset.addEventListener('click', () => {
  if (!ws || ws.readyState !== WebSocket.OPEN) return showToast('Not connected');
  ws.send(JSON.stringify({ type: 'reset', prompt: promptEl.value || null }));
  fpsHistory = [];
  drawSparkline(fpsSpark, fpsHistory);
  showToast('Reset');
});
btnDownload.addEventListener('click', () => {
  const a = document.createElement('a');
  a.href = canvas.toDataURL('image/png');
  a.download = `frame_${Date.now()}.png`;
  a.click();
});
btnRecord.addEventListener('click', () => {
  if (!('MediaRecorder' in window)) { showToast('Recording not supported'); return; }
  if (!recording) startRecording(); else stopRecording();
});

function startRecording() {
  try {
    const stream = canvas.captureStream(30);
    const mime = MediaRecorder.isTypeSupported('video/webm;codecs=vp9') ? 'video/webm;codecs=vp9' : 'video/webm';
    mediaRecorder = new MediaRecorder(stream, { mimeType: mime, videoBitsPerSecond: 3_000_000 });
    recordedChunks = [];
    mediaRecorder.ondataavailable = (e) => { if (e.data.size > 0) recordedChunks.push(e.data); };
    mediaRecorder.onstop = () => {
      const blob = new Blob(recordedChunks, { type: mediaRecorder.mimeType });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url; a.download = `lwm_${Date.now()}.webm`; a.click();
      URL.revokeObjectURL(url);
    };
    mediaRecorder.start();
    recording = true;
    btnRecord.textContent = 'Stop';
    recBadge.classList.remove('hidden');
    showToast('Recording…');
  } catch (e) {
    console.error(e); showToast('Failed to start recording');
  }
}

function stopRecording() {
  try {
    mediaRecorder.stop();
    recording = false;
    btnRecord.textContent = 'Record';
    recBadge.classList.add('hidden');
    showToast('Saved recording');
  } catch (e) {
    console.error(e); showToast('Failed to stop recording');
  }
}
btnCopy.addEventListener('click', async () => {
  if (!sessionId) return;
  await navigator.clipboard.writeText(sessionId);
  showToast('Session ID copied');
});

document.addEventListener('keydown', (e) => {
  const key = e.key.toLowerCase();
  if (actionMap.hasOwnProperty(key) && !pressedKeys.has(key)) {
    pressedKeys.add(key);
    sendAction(actionMap[key]);
    const el = document.querySelector(`[data-key="${key}"]`);
    if (el) el.classList.add('active');
  }
});
document.addEventListener('keyup', (e) => {
  const key = e.key.toLowerCase();
  pressedKeys.delete(key);
  const el = document.querySelector(`[data-key="${key}"]`);
  if (el) el.classList.remove('active');
});

document.querySelectorAll('.dpad-btn').forEach((btn) => {
  btn.addEventListener('click', () => {
    const act = btn.getAttribute('data-action');
    const key = act === 'up' ? 'w' : act === 'left' ? 'a' : act === 'down' ? 's' : act === 'right' ? 'd' : ' ';
    sendAction(actionMap[key]);
  });
});

// Theme toggle
const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
const THEME_KEY = 'lwm-theme';
function applyTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme);
}
const savedTheme = localStorage.getItem(THEME_KEY) || (prefersDark ? 'dark' : 'dark');
applyTheme(savedTheme);
themeToggle.addEventListener('click', () => {
  const current = document.documentElement.getAttribute('data-theme') || 'dark';
  const next = current === 'dark' ? 'light' : 'dark';
  applyTheme(next); localStorage.setItem(THEME_KEY, next);
});

// Initial draw sparkline
drawSparkline(fpsSpark, fpsHistory);
