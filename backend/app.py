"""
FastAPI backend — Squat Analyzer API.

Endpoints:
  POST /analyze    — Upload video for batch analysis.
  WS   /realtime   — Low-latency live webcam analysis.

Latency design for /realtime
-----------------------------
The server does NOT send annotated JPEG frames back to the client.
Instead it returns only JSON: keypoints + metrics (~2KB vs ~50KB for JPEG).

The frontend draws the raw webcam feed locally at 30fps via requestAnimationFrame
and overlays the skeleton from the JSON keypoints. Server round-trip only
affects the skeleton overlay, not video smoothness.

Additional backpressure: the send loop in the frontend waits for a response
before sending the next frame, so inference always runs on the freshest frame
rather than a growing queue of stale ones.
"""

import asyncio
import base64
import os
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.websockets import WebSocket, WebSocketDisconnect
from mediapipe.tasks.python import vision

from pose_estimator import PoseEstimator
from squat_analyzer import analyze_squat
from feature_extractor import PoseLandmarks, FrameFeatures

app = FastAPI(title="Squat Analyzer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Thread pool for CPU-bound MediaPipe inference — keeps async loop unblocked
_executor = ThreadPoolExecutor(max_workers=2)


# ---------------------------------------------------------------------------
# POST /analyze — video file upload
# ---------------------------------------------------------------------------

@app.post("/analyze")
async def analyze_video(file: UploadFile):
    suffix = os.path.splitext(file.filename or ".mp4")[1] or ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    video_path = tmp.name
    try:
        shutil.copyfileobj(file.file, tmp)
        tmp.close()
        loop   = asyncio.get_event_loop()
        result = await loop.run_in_executor(_executor, analyze_squat, video_path)
        return result
    finally:
        try:
            tmp.close()
            os.remove(video_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# WS /realtime — keypoints + metrics JSON only, no annotated frame
# ---------------------------------------------------------------------------

def _extract_frame_data(frame: np.ndarray,
                        estimator: PoseEstimator,
                        timestamp_ms: int) -> dict:
    """
    Run pose estimation on one frame.
    Returns JSON-serialisable dict with keypoints and metrics.
    No JPEG encoding — caller draws the skeleton client-side.
    Pure synchronous — runs in thread pool.
    """
    raw_kp = estimator.extract_keypoints_from_frame(frame, timestamp_ms)
    if raw_kp is None:
        return {"detected": False}

    lm = PoseLandmarks(raw_kp)
    ff = FrameFeatures.from_landmarks(lm)

    # Send only (x, y, visibility) per landmark — z not needed for 2D overlay
    keypoints = [
        {"x": round(kp[0], 4), "y": round(kp[1], 4), "v": round(kp[3], 2)}
        for kp in raw_kp
    ]

    metrics = {}
    if ff.left_knee_angle  is not None: metrics["left_knee_angle"]  = round(ff.left_knee_angle, 1)
    if ff.right_knee_angle is not None: metrics["right_knee_angle"] = round(ff.right_knee_angle, 1)
    if ff.left_hip_angle   is not None: metrics["hip_angle"]        = round(ff.left_hip_angle, 1)
    if ff.back_angle       is not None: metrics["back_angle"]       = round(ff.back_angle, 1)

    valgus = bool(
        (ff.left_knee_valgus_ratio  is not None and ff.left_knee_valgus_ratio  > 0.15) or
        (ff.right_knee_valgus_ratio is not None and ff.right_knee_valgus_ratio > 0.15)
    )
    metrics["faults"] = {
        "knee_valgus":  valgus,
        "forward_lean": bool(ff.back_angle is not None and ff.back_angle > 45),
    }

    return {
        "detected": True,
        "keypoints": keypoints,
        "metrics":   metrics,
        "text":      _frame_feedback(metrics),
    }


@app.websocket("/realtime")
async def realtime_feedback(websocket: WebSocket):
    """
    Receive base64 JPEG frames from the client.
    Run pose estimation in a thread pool.
    Return only JSON (keypoints + metrics) — no JPEG frame back.

    Backpressure: latest_frame is a single-slot buffer. While inference
    is running, incoming frames overwrite it so we always process the
    newest frame, never a queue of stale ones.
    """
    await websocket.accept()

    estimator    = PoseEstimator(running_mode=vision.RunningMode.VIDEO)
    start_time   = time.monotonic()
    loop         = asyncio.get_event_loop()
    processing   = False
    latest_frame = [None]  # single-slot buffer

    try:
        while True:
            # Receive next frame from client
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=5.0)
            except asyncio.TimeoutError:
                continue

            # Decode base64 JPEG → numpy BGR
            try:
                encoded   = data.split(",", 1)[1] if "," in data else data
                img_bytes = base64.b64decode(encoded)
                nparr     = np.frombuffer(img_bytes, np.uint8)
                frame     = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            except Exception:
                continue

            if frame is None:
                continue

            # Buffer the frame; if inference is busy, just update the slot
            latest_frame[0] = frame
            if processing:
                continue

            # Drain the buffer — process the latest frame
            processing = True
            while latest_frame[0] is not None:
                f              = latest_frame[0]
                latest_frame[0] = None
                ts             = int((time.monotonic() - start_time) * 1000)

                result = await loop.run_in_executor(
                    _executor, _extract_frame_data, f, estimator, ts
                )

                await websocket.send_json(result)

            processing = False

    except (WebSocketDisconnect, Exception) as e:
        print(f"WebSocket closed: {e}")
    finally:
        estimator.close()


def _frame_feedback(metrics: dict) -> str:
    if not metrics:
        return "No pose detected — make sure your full body is visible."
    faults = metrics.get("faults", {})
    if faults.get("knee_valgus"):
        return "Push your knees out!"
    if faults.get("forward_lean"):
        return "Keep your chest up!"
    lka = metrics.get("left_knee_angle", 180)
    rka = metrics.get("right_knee_angle", 180)
    if min(lka, rka) > 100:
        return "Squat deeper — aim for parallel."
    return "Good form — keep it up!"