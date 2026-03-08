# SquatAI — Squat Form Analyzer

An AI-powered squat grading system that extracts pose keypoints from video, computes biomechanical angles, detects movement faults, and generates actionable coaching feedback.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start the backend
cd backend
uvicorn app:app --reload --port 8000

# 3. Open the frontend
open frontend/index.html
# (or serve it: python -m http.server 3000 --directory frontend)
```

---

## Architecture

```
frontend/
  index.html      — Single-page UI (upload + live tabs)
  app.js          — Fetch/WebSocket client logic

backend/
  app.py              — FastAPI: POST /analyze, WS /realtime
  pose_estimator.py   — MediaPipe PoseLandmarker wrapper
  squat_analyzer.py   — Full pipeline (batch + per-frame)
  biomechanics.py     — 3D joint angle math
  fault_detection.py  — Knee valgus, heel lift, asymmetry, forward lean
  scoring.py          — Heuristic rule-based scoring (1–10)
  ai_coach.py         — Feedback string generation from faults
  smoothing.py        — Savitzky-Golay temporal smoothing
  rep_counter.py      — Peak detection on hip y-position
  movement_quality.py — Velocity variance consistency score
  visualizer.py       — OpenCV landmark + angle overlay
  ml_model.py         — PyTorch SquatNet (untrained scaffold)
  models/
    pose_landmarker.task  — MediaPipe full model weights
```

---

## Model Choice

**MediaPipe PoseLandmarker (Full model)**

MediaPipe was chosen over alternatives (OpenPose, MoveNet, HRNet) for these reasons:

| Criteria | MediaPipe Full | MoveNet Thunder | HRNet |
|---|---|---|---|
| Inference speed | ~30ms/frame | ~15ms | ~100ms |
| Landmark count | 33 (+ world coords) | 17 | 17 |
| 3D coordinates | ✅ (x, y, z) | ❌ | ❌ |
| CPU-only viable | ✅ | ✅ | ❌ |
| License | Apache 2.0 | Apache 2.0 | Research only |

The full model (vs lite) provides better accuracy on partially occluded poses — important when athletes face sideways or at angles to the camera.

**Why not OpenPose?** GPL license, slower, requires GPU for real-time use.

**Why not HRNet?** Research license restricts commercial use; no CPU-viable inference path.

---

## Joint Angle Computation

All angles use the 3D vector dot-product formula to avoid gimbal lock issues that affect 2D-only approaches:

```
angle(a, b, c) = arccos( (a-b)·(c-b) / (|a-b| * |c-b|) )
```

Key joints extracted per frame:

| Metric | Landmarks Used |
|---|---|
| Knee angle | Hip → Knee → Ankle |
| Hip angle | Shoulder → Hip → Knee |
| Back angle | Shoulder–Hip vector vs vertical |
| Squat depth | Minimum knee angle across all frames |

Temporal smoothing uses a **Savitzky-Golay filter** (window=7, poly=2), which preserves angle curve peaks (squat bottom) better than a simple moving average.

---

## Scoring Logic

The current scorer is **heuristic / rule-based** (`scoring.py`):

```
score = 10
  − 3  if min knee angle > 100° (insufficient depth)
  − 3  if max back angle > 45°  (excessive forward lean)
  − 2  if min hip angle > 90°   (hip not below parallel)
floor at 1
```

A PyTorch `SquatNet` scaffold exists in `ml_model.py` (4 features → 1 score) but is **untrained** — labeled training data is required before activating it. The heuristic scorer is production-safe; the ML model is a demonstration of the intended pipeline upgrade path.

---

## Fault Detection

| Fault | Method |
|---|---|
| Knee valgus | Knee lateral deviation from hip–ankle line > 8cm (normalized) |
| Heel lift | Ankle y-variance > 0.05 across the rep |
| Forward lean | Max back angle > 50° |
| L/R asymmetry | Difference in L vs R minimum knee angle > 10° |

Valgus detection is evaluated at the **deepest squat frame** (minimum average knee angle) rather than the last frame — this avoids false negatives from the standing position.

---

## Scaling Plan

### Phase 1 — Current (prototype)
- Single-user, local FastAPI server
- CPU inference via MediaPipe
- Heuristic scoring

### Phase 2 — Production MVP
- Containerize backend with Docker
- Deploy to a GPU-enabled instance (e.g. AWS g4dn.xlarge) for <10ms inference
- Add S3 presigned URL video upload (skip streaming large files through the API)
- PostgreSQL for storing analysis history per user
- Auth via JWT

### Phase 3 — Mobile + Edge
- Export pose model to TFLite / CoreML for on-device inference
- React Native frontend consuming the same REST API
- Edge inference means zero video upload latency

### Phase 4 — Volleyball Drill Extension
- Swap squat keypoint logic for drill-specific joint sequences (approach jump, arm swing)
- Train drill classifier on labeled video dataset
- Per-drill scoring models fine-tuned with labeled coach feedback

---

## Limitations

1. **Single person only** — MediaPipe returns the first detected person; multi-athlete scenes are not supported.
2. **Camera angle matters** — Sagittal (side) view gives the most accurate knee/hip angles. Front view cannot measure sagittal plane depth accurately.
3. **Occlusion** — Baggy clothing, bands, or partial frame occlusion reduces landmark confidence and can corrupt angle readings.
4. **3D angles are still projected** — MediaPipe's z-coordinate is estimated, not measured (no depth sensor). Angles are more accurate than 2D-only but not equivalent to markerless motion capture.
5. **Heuristic thresholds are not personalized** — A 90° knee angle standard doesn't account for individual anatomy (femur length, hip structure). Personalization requires per-user calibration.
6. **Untrained ML model** — `SquatNet` in `ml_model.py` is a scaffold only and must not be used for scoring without a labeled training dataset.
7. **No temporal context in real-time mode** — The WebSocket endpoint analyzes each frame independently; rep boundaries and phase detection require multi-frame buffering (planned for v2).

---

## Feature Extraction Framework

The feature extraction is structured in three explicit layers:

```
PoseLandmarks         — named access (lm.left_knee, not kp[25])
    ↓
FrameFeatures         — all biomechanics for one frame (angles, offsets, heights)
    ↓
SquatFeatures         — temporally aggregated rep-level feature set
    ↓
ScoreBreakdown        — per-dimension weighted scores → total (1-10)
```

**`PoseLandmarks`** — eliminates magic index numbers everywhere. Single source of truth for MediaPipe indices (`MP.LEFT_KNEE = 25`). Visibility-gated: returns `None` for occluded landmarks rather than silently using bad data.

**`FrameFeatures`** — computed via `FrameFeatures.from_landmarks(lm)`. Contains every measurable value per frame: knee angles, hip angles, back angle, ankle dorsiflexion, lateral knee offset, and positional values. All values are `Optional[float]` — callers handle None explicitly.

**`SquatFeatures`** — produced by `FeatureExtractor.aggregate(frames)`. Smooths signals, detects phases, sets fault flags, and exposes `to_ml_vector()` for model input and `to_dict()` for the API response.

**`ScoreBreakdown`** — five independently-scored dimensions (depth, posture, symmetry, heel contact, consistency) combined via configurable weights in `scoring.py`.

---

## Volleyball Drill Architecture Roadmap

The framework above was designed to extend to volleyball drills with minimal new code.

### Core Principle

A volleyball drill is just a different **keypoint sequence pattern** evaluated against different **biomechanical rules**. The `PoseLandmarks → FrameFeatures → DrillFeatures → ScoreBreakdown` pipeline stays identical — only the feature definitions and scoring weights change.

### Planned Drill Modules

#### 1. Approach Jump (`drills/approach_jump.py`)
Analyzes the 4-step approach and takeoff.

Key features to extract:
- Arm swing velocity (wrist y-displacement over penultimate and last steps)
- Hip-to-shoulder rotation angle at plant
- Knee bend depth at penultimate step (~100-110° optimal)
- Vertical jump height (estimated from hip apex height)
- Takeoff symmetry (bilateral foot timing)

Phases: `approach → plant → arm_load → takeoff → flight`

#### 2. Blocking (`drills/blocking.py`)
Analyzes lateral footwork and vertical penetration.

Key features:
- Lateral shuffle step width and timing
- Arm extension angle above net height (wrist y at peak)
- Elbow extension at block contact
- Hand spread (wrist distance at contact)

#### 3. Passing / Digging (`drills/passing.py`)
Analyzes platform angle and body positioning.

Key features:
- Forearm platform angle relative to target direction
- Hip hinge depth at contact point
- Knee bend (low platform = deeper bend)
- Shoulder alignment to ball trajectory

### Architecture for Adding a Drill

```python
# 1. Create DrillFrameFeatures subclass
class ApproachJumpFrameFeatures(FrameFeatures):
    arm_swing_velocity: Optional[float] = None
    hip_rotation_angle: Optional[float] = None

    @classmethod
    def from_landmarks(cls, lm: PoseLandmarks) -> "ApproachJumpFrameFeatures":
        f = super().from_landmarks(lm)   # inherit base features
        # add drill-specific calculations
        ...
        return f

# 2. Create DrillFeatures aggregator
class ApproachJumpFeatures(SquatFeatures):
    max_arm_swing_velocity: float = 0.0
    jump_height_estimate:   float = 0.0
    ...

# 3. Register drill weights in scoring.py
DRILL_WEIGHTS = {
    "approach_jump": {
        "arm_swing":   0.30,
        "jump_height": 0.25,
        "knee_bend":   0.25,
        "symmetry":    0.20,
    }
}

# 4. Add API endpoint
@app.post("/analyze/{drill}")
async def analyze_drill(drill: str, file: UploadFile): ...
```

### Phase Detection for Drills

`movement_phases.py` provides the foundation. Each drill needs its own phase detector:

```python
# Squat phases: standing → descent → bottom → ascent → standing
# Approach jump phases: stride1 → stride2 → penultimate → plant → flight → land

def detect_approach_phases(hip_positions, ankle_velocities):
    # Use peaks in ankle velocity to detect each step
    ...
```

Phase-aware features (e.g., "arm swing angle *at* takeoff") are far more informative than global min/max values, and the framework supports this via frame index ranges returned by phase detectors.

### Multi-Person Support (Team Drills)

MediaPipe PoseLandmarker supports multiple people when configured:

```python
options = vision.PoseLandmarkerOptions(
    ...
    num_poses=6   # track full 6-person rotation
)
```

Each detected person returns a separate landmark list. The pipeline processes each athlete independently and can compare metrics across the team in a single pass.

### Mobile / Edge Deployment

The `PoseLandmarker` model comes in three sizes:
- `pose_landmarker_lite.task` — fastest, suitable for real-time on-device (iOS/Android)
- `pose_landmarker_full.task` — current default, good balance
- `pose_landmarker_heavy.task` — highest accuracy, server-side only

For mobile, swap to the lite model and run inference via TFLite on-device. The feature extraction and scoring logic runs identically on the server — only the model file and inference call change.
