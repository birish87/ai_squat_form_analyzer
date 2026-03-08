/**
 * SquatAI — Frontend
 *
 * Live analysis architecture:
 *   Webcam → MediaPipe JS (in-browser, zero network) → canvas overlay + angle UI
 *
 * The Python backend is only used for the full video upload analysis.
 * All real-time pose estimation runs locally via the MediaPipe Tasks JS SDK,
 * which eliminates the encode→send→infer→encode→receive round-trip entirely.
 */

const API_BASE = "http://localhost:8000";

// ── Tab nav ───────────────────────────────────────────────────────────────

document.querySelectorAll(".tab-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".panel").forEach(p => p.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(`panel-${btn.dataset.tab}`).classList.add("active");
    if (btn.dataset.tab !== "live" && liveActive) stopLive();
  });
});

// ── Upload / Analyze ──────────────────────────────────────────────────────

const videoInput     = document.getElementById("videoUpload");
const analyzeBtn     = document.getElementById("analyzeBtn");
const fileNameEl     = document.getElementById("file-name");
const loader         = document.getElementById("loader");
const emptyState     = document.getElementById("emptyState");
const resultsContent = document.getElementById("resultsContent");
const dropZone       = document.getElementById("dropZone");

videoInput.addEventListener("change", () => {
  const f = videoInput.files[0];
  if (f) { fileNameEl.textContent = `✓ ${f.name}`; analyzeBtn.disabled = false; }
});

dropZone.addEventListener("dragover",  e => { e.preventDefault(); dropZone.classList.add("drag-over"); });
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
dropZone.addEventListener("drop", e => {
  e.preventDefault(); dropZone.classList.remove("drag-over");
  const f = e.dataTransfer.files[0];
  if (f && f.type.startsWith("video/")) {
    const dt = new DataTransfer(); dt.items.add(f); videoInput.files = dt.files;
    fileNameEl.textContent = `✓ ${f.name}`; analyzeBtn.disabled = false;
  }
});

async function uploadVideo() {
  const file = videoInput.files[0];
  if (!file) return;
  emptyState.style.display = "none";
  resultsContent.style.display = "none";
  loader.classList.add("show");
  analyzeBtn.disabled = true;

  const fd = new FormData();
  fd.append("file", file);
  try {
    const res = await fetch(`${API_BASE}/analyze`, { method: "POST", body: fd });
    if (!res.ok) throw new Error(`Server error ${res.status}: ${await res.text()}`);
    renderResults(await res.json());
  } catch (err) {
    renderError(err.message);
  } finally {
    loader.classList.remove("show");
    analyzeBtn.disabled = false;
  }
}

function renderResults(data) {
  const { score, metrics, feedback } = data;
  if (!score && score !== 0) { renderError("No response data received."); return; }
  const pct = score * 10;
  const label = score >= 8 ? "Excellent" : score >= 6 ? "Good" : score >= 4 ? "Needs Work" : "Poor";
  const rows = [
    { label: "Min Knee Angle", value: metrics.min_knee_angle != null ? `${metrics.min_knee_angle.toFixed(1)}°` : "—", warn: metrics.min_knee_angle > 100 },
    { label: "Min Hip Angle",  value: metrics.min_hip_angle  != null ? `${metrics.min_hip_angle.toFixed(1)}°`  : "—", warn: metrics.min_hip_angle  > 90  },
    { label: "Max Back Angle", value: metrics.max_back_angle != null ? `${metrics.max_back_angle.toFixed(1)}°` : "—", warn: metrics.max_back_angle > 45  },
    { label: "Squat Depth",    value: metrics.squat_depth  || "—",  warn: metrics.squat_depth === "above parallel" },
    { label: "Reps Detected",  value: metrics.rep_count    != null ? metrics.rep_count : "—", warn: false },
    { label: "Consistency",    value: metrics.movement_consistency != null ? metrics.movement_consistency.toFixed(1) : "—", warn: metrics.movement_consistency < 5 },
  ];
  resultsContent.innerHTML = `
    <div class="score-row">
      <div class="score-dial" style="--pct:${pct}"><span class="score-num">${score}</span></div>
      <div>
        <div class="score-label">${label}</div>
        <div class="score-sub">out of 10 — ${metrics.rep_count || 0} rep(s) analyzed</div>
      </div>
    </div>
    <div class="metrics-grid">${rows.map(m => `
      <div class="metric-pill ${m.warn ? "warn" : "ok"}">
        <div class="label">${m.label}</div><div class="value">${m.value}</div>
      </div>`).join("")}
    </div>
    <div class="feedback-title">Actionable Suggestions</div>
    <ul class="feedback-list">${(feedback||[]).map(f => `<li>${f}</li>`).join("") || "<li>No issues detected — great squat!</li>"}</ul>`;
  resultsContent.style.display = "block";
}

function renderError(msg) {
  resultsContent.innerHTML = `<div style="color:var(--accent2);font-family:var(--mono);font-size:0.82rem;line-height:1.6">⚠ ${msg}</div>`;
  resultsContent.style.display = "block";
}

// ── In-browser MediaPipe live analysis ───────────────────────────────────
//
// Uses @mediapipe/tasks-vision loaded from CDN.
// Pose estimation runs entirely in the browser via WASM — no server round-trip.

let poseLandmarker = null;
let liveActive     = false;
let rafId          = null;
let stream         = null;

// FPS tracking
let fpsFrameCount  = 0;
let fpsLastTime    = performance.now();

const canvas      = document.getElementById("liveCanvas");
const ctx         = canvas.getContext("2d");
const liveDot     = document.getElementById("liveDot");
const liveStatus  = document.getElementById("liveStatusText");
const liveBtn     = document.getElementById("liveBtn");
const liveCue     = document.getElementById("liveCue");
const fpsBadge    = document.getElementById("fpsBadge");
const modelStatus = document.getElementById("modelStatus");

// MediaPipe connection colours
const LANDMARK_COLOR     = "#00ff88";
const CONNECTION_COLOR   = "#00ccff";
const ANGLE_COLOR        = "#c8f135";
const ANGLE_WARN_COLOR   = "#ff5c35";

// MediaPipe pose connections (pairs of indices)
const CONNECTIONS = [
  [11,12],[11,13],[13,15],[12,14],[14,16],
  [11,23],[12,24],[23,24],
  [23,25],[25,27],[27,29],[27,31],
  [24,26],[26,28],[28,30],[28,32],
];

// Load MediaPipe Tasks Vision from CDN
async function loadModel() {
  try {
    const { PoseLandmarker, FilesetResolver } =
      await import("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs");

    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
    );

    poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
        delegate: "GPU",
      },
      runningMode: "VIDEO",
      numPoses: 1,
    });

    modelStatus.textContent = "Model ready ✓";
    modelStatus.className = "model-status ready";
    liveBtn.disabled = false;
    liveBtn.textContent = "Start Live Analysis";
  } catch (err) {
    modelStatus.textContent = `Model failed: ${err.message}`;
    modelStatus.className = "model-status error";
    console.error("MediaPipe load error:", err);
  }
}

loadModel();

// ── Live loop ─────────────────────────────────────────────────────────────

function toggleLive() { liveActive ? stopLive() : startLive(); }

async function startLive() {
  if (!poseLandmarker) return;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
  } catch (e) {
    alert("Could not access webcam: " + e.message); return;
  }

  // Hidden video element for webcam frames
  const video = document.createElement("video");
  video.srcObject = stream;
  video.autoplay = true;
  video.muted = true;
  video.playsInline = true;
  await video.play();

  liveActive = true;
  liveDot.classList.add("live");
  liveStatus.textContent = "Live";
  liveBtn.textContent = "Stop";
  liveBtn.classList.add("stop");
  liveCue.classList.add("show");
  fpsLastTime = performance.now();
  fpsFrameCount = 0;

  let lastVideoTime = -1;

  function loop(now) {
    if (!liveActive) return;
    rafId = requestAnimationFrame(loop);

    if (video.readyState < 2) return;          // frame not ready yet
    if (video.currentTime === lastVideoTime) return;  // same frame, skip
    lastVideoTime = video.currentTime;

    // Draw raw webcam frame first
    canvas.width  = video.videoWidth  || 640;
    canvas.height = video.videoHeight || 480;
    ctx.drawImage(video, 0, 0);

    // Run pose estimation on this frame
    const result = poseLandmarker.detectForVideo(video, now);

    if (result.landmarks && result.landmarks.length > 0) {
      const lms = result.landmarks[0];

      // Draw skeleton
      drawSkeleton(lms);

      // Compute angles
      const metrics = computeMetrics(lms);

      // Draw angle labels
      drawAngleLabel(lms[25], metrics.leftKnee,  canvas.width, canvas.height, "knee");
      drawAngleLabel(lms[26], metrics.rightKnee, canvas.width, canvas.height, "knee");
      drawAngleLabel(lms[23], metrics.leftHip,   canvas.width, canvas.height, "hip");

      // Update sidebar
      updateUI(metrics);
    } else {
      liveCue.textContent = "No pose — step back so full body is visible";
      resetAngleBars();
    }

    // FPS counter
    fpsFrameCount++;
    if (now - fpsLastTime >= 1000) {
      fpsBadge.textContent = `${fpsFrameCount} fps`;
      fpsFrameCount = 0;
      fpsLastTime = now;
    }
  }

  rafId = requestAnimationFrame(loop);
}

function stopLive() {
  liveActive = false;
  if (rafId) { cancelAnimationFrame(rafId); rafId = null; }
  if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  liveDot.classList.remove("live");
  liveStatus.textContent = "Camera off";
  liveBtn.textContent = "Start Live Analysis";
  liveBtn.classList.remove("stop");
  liveCue.textContent = "Start to begin";
  fpsBadge.textContent = "— fps";
  resetAngleBars();
}

// ── Drawing ───────────────────────────────────────────────────────────────

function lmPx(lm) {
  return [lm.x * canvas.width, lm.y * canvas.height];
}

function drawSkeleton(lms) {
  // Connections
  ctx.strokeStyle = CONNECTION_COLOR;
  ctx.lineWidth = 2;
  ctx.globalAlpha = 0.7;
  for (const [a, b] of CONNECTIONS) {
    if (lms[a].visibility < 0.4 || lms[b].visibility < 0.4) continue;
    const [ax, ay] = lmPx(lms[a]);
    const [bx, by] = lmPx(lms[b]);
    ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke();
  }
  // Dots
  ctx.globalAlpha = 1;
  for (const lm of lms) {
    if (lm.visibility < 0.4) continue;
    const [x, y] = lmPx(lm);
    ctx.beginPath(); ctx.arc(x, y, 4, 0, Math.PI * 2);
    ctx.fillStyle = LANDMARK_COLOR; ctx.fill();
    ctx.strokeStyle = "#000"; ctx.lineWidth = 1; ctx.stroke();
  }
}

const ANGLE_COLORS = { good: "#00e676", caution: "#ffcc00", bad: "#ff5c35" };

function drawAngleLabel(lm, angle, w, h, type = "knee") {
  if (angle == null || lm.visibility < 0.4) return;
  const x = lm.x * w + 12;
  const y = lm.y * h + 6;
  const label = `${Math.round(angle)}°`;
  const color = ANGLE_COLORS[angleClass(angle, type)];
  ctx.font = "bold 14px 'DM Mono', monospace";
  ctx.fillStyle = "rgba(0,0,0,0.6)";
  const tw = ctx.measureText(label).width;
  ctx.fillRect(x - 3, y - 14, tw + 6, 18);
  ctx.fillStyle = color;
  ctx.fillText(label, x, y);
}

// ── Biomechanics (JS port of biomechanics.py) ─────────────────────────────

function vec3(lm) { return [lm.x, lm.y, lm.z]; }

function angleBetween(a, b, c) {
  const ba = [a[0]-b[0], a[1]-b[1], a[2]-b[2]];
  const bc = [c[0]-b[0], c[1]-b[1], c[2]-b[2]];
  const dot = ba[0]*bc[0] + ba[1]*bc[1] + ba[2]*bc[2];
  const magBa = Math.sqrt(ba[0]**2 + ba[1]**2 + ba[2]**2);
  const magBc = Math.sqrt(bc[0]**2 + bc[1]**2 + bc[2]**2);
  if (magBa === 0 || magBc === 0) return null;
  return (Math.acos(Math.max(-1, Math.min(1, dot / (magBa * magBc)))) * 180) / Math.PI;
}

function backAngle(shoulder, hip) {
  const torso   = [shoulder[0]-hip[0], shoulder[1]-hip[1], shoulder[2]-hip[2]];
  const vertical = [0, 1, 0];
  const dot  = torso[1];  // dot with [0,1,0] = just y component
  const mag  = Math.sqrt(torso[0]**2 + torso[1]**2 + torso[2]**2);
  if (mag === 0) return null;
  return (Math.acos(Math.max(-1, Math.min(1, dot / mag))) * 180) / Math.PI;
}

function valgusRatio(hip, knee, ankle) {
  // JS port of feature_extractor.py valgus projection logic
  const hipAnkleDy = ankle[1] - hip[1];
  if (Math.abs(hipAnkleDy) < 1e-4) return 0;
  const t = (knee[1] - hip[1]) / hipAnkleDy;
  const expectedX = hip[0] + t * (ankle[0] - hip[0]);
  const dist = Math.sqrt((ankle[0]-hip[0])**2 + (ankle[1]-hip[1])**2);
  if (dist < 1e-4) return 0;
  return (expectedX - knee[0]) / dist;  // left leg: positive = valgus
}

function computeMetrics(lms) {
  const lh = vec3(lms[23]), rh = vec3(lms[24]);
  const lk = vec3(lms[25]), rk = vec3(lms[26]);
  const la = vec3(lms[27]), ra = vec3(lms[28]);
  const ls = vec3(lms[11]), rs = vec3(lms[12]);
  const mh = [(lh[0]+rh[0])/2, (lh[1]+rh[1])/2, (lh[2]+rh[2])/2];
  const ms = [(ls[0]+rs[0])/2, (ls[1]+rs[1])/2, (ls[2]+rs[2])/2];

  const leftKnee   = lms[25].visibility > 0.4 ? angleBetween(lh, lk, la) : null;
  const rightKnee  = lms[26].visibility > 0.4 ? angleBetween(rh, rk, ra) : null;
  const leftHip    = lms[23].visibility > 0.4 ? angleBetween(ls, lh, lk) : null;
  const back       = backAngle(ms, mh);

  const leftValgus  = lms[25].visibility > 0.4 ? valgusRatio(lh, lk, la) : 0;
  const rightValgus = lms[26].visibility > 0.4 ? -valgusRatio(rh, rk, ra) : 0; // right leg inverted

  return {
    leftKnee, rightKnee, leftHip, back,
    faultValgus:  leftValgus > 0.15 || rightValgus > 0.15,
    faultForward: back != null && back > 45,
  };
}

// ── Sidebar UI ────────────────────────────────────────────────────────────

function updateUI(m) {
  setAngleBar("lkaVal", "lkaBar", m.leftKnee,  180, "knee");
  setAngleBar("rkaVal", "rkaBar", m.rightKnee, 180, "knee");
  setAngleBar("hipVal", "hipBar", m.leftHip,   180, "hip");
  setAngleBar("backVal","backBar", m.back,       90, "back");

  setFault("fault-valgus", m.faultValgus);
  setFault("fault-lean",   m.faultForward);

  // Coaching cue
  if (m.faultValgus)       liveCue.textContent = "Push your knees out!";
  else if (m.faultForward) liveCue.textContent = "Keep your chest up!";
  else if (m.leftKnee != null && m.leftKnee > 100 &&
           m.rightKnee != null && m.rightKnee > 100) liveCue.textContent = "Squat deeper — aim for parallel.";
  else                     liveCue.textContent = "Good form — keep it up!";
}

// Thresholds per metric — [good_max, caution_max]
const ANGLE_THRESHOLDS = {
  knee: [100, 120],   // good ≤100, caution ≤120, bad >120
  hip:  [100, 120],
  back: [25,  45],    // good ≤25,  caution ≤45,  bad >45
};

function angleClass(angle, type) {
  const [g, c] = ANGLE_THRESHOLDS[type] || [100, 120];
  if (angle <= g) return "good";
  if (angle <= c) return "caution";
  return "bad";
}

function setAngleBar(valId, barId, angle, max = 180, type = "knee") {
  const el  = document.getElementById(valId);
  const bar = document.getElementById(barId);
  if (angle != null) {
    el.textContent = `${Math.round(angle)}°`;
    bar.style.width = `${Math.min(100, (angle / max) * 100)}%`;
    const cls = angleClass(angle, type);
    // Apply colour class to bar
    bar.className = `angle-bar-fill ${cls}`;
    // Apply colour class to value text
    el.className  = `angle-val ${cls}`;
  }
}

function setFault(id, active) {
  const el = document.getElementById(id);
  el.classList.toggle("detected", !!active);
  el.classList.toggle("clear",    !active);
}

function resetAngleBars() {
  ["lkaVal","rkaVal","hipVal","backVal"].forEach(id => {
    const el = document.getElementById(id);
    el.textContent = "—°";
    el.className = "angle-val";
  });
  ["lkaBar","rkaBar","hipBar","backBar"].forEach(id => {
    const el = document.getElementById(id);
    el.style.width = "0%";
    el.className = "angle-bar-fill";
  });
  setFault("fault-valgus", false);
  setFault("fault-lean",   false);
}