"""
Squat analysis pipeline — wired to the feature extraction framework.

Flow:
  video/frame
    → PoseEstimator        (keypoints)
    → PoseLandmarks        (named access)
    → FrameFeatures        (per-frame biomechanics)
    → FeatureExtractor     (temporal aggregation → SquatFeatures)
    → score_squat          (ScoreBreakdown)
    → generate_feedback    (coaching cues)
"""

import cv2
import numpy as np
from mediapipe.tasks.python import vision

from pose_estimator import PoseEstimator
from feature_extractor import PoseLandmarks, FrameFeatures, FeatureExtractor
from scoring import score_squat
from ai_coach import generate_feedback
from visualizer import draw_landmarks, draw_angle


# ---------------------------------------------------------------------------
# Single-frame analysis  (WebSocket real-time endpoint)
# ---------------------------------------------------------------------------

def analyze_squat_frame(frame: np.ndarray,
                        pose_estimator: PoseEstimator,
                        timestamp_ms: int = 0):
    """
    Analyze one frame. Returns (annotated_frame, metrics_dict).
    metrics_dict is empty if no pose detected.
    """
    raw_kp = pose_estimator.extract_keypoints_from_frame(frame, timestamp_ms)
    if raw_kp is None:
        return frame, {}

    lm = PoseLandmarks(raw_kp)
    ff = FrameFeatures.from_landmarks(lm)

    metrics = {}
    if ff.left_knee_angle  is not None: metrics["left_knee_angle"]  = round(ff.left_knee_angle, 1)
    if ff.right_knee_angle is not None: metrics["right_knee_angle"] = round(ff.right_knee_angle, 1)
    if ff.left_hip_angle   is not None: metrics["hip_angle"]        = round(ff.left_hip_angle, 1)
    if ff.back_angle       is not None: metrics["back_angle"]       = round(ff.back_angle, 1)

    metrics["faults"] = {
        "knee_valgus":  bool(
            (ff.left_knee_valgus_ratio  is not None and ff.left_knee_valgus_ratio  > 0.15) or
            (ff.right_knee_valgus_ratio is not None and ff.right_knee_valgus_ratio > 0.15)
        ),
        "forward_lean": bool(ff.back_angle is not None and ff.back_angle > 45),
    }

    # Annotate frame
    frame = draw_landmarks(frame, raw_kp)
    if lm.left_hip and lm.left_knee and lm.left_ankle and ff.left_knee_angle:
        frame = draw_angle(frame, lm.left_hip, lm.left_knee, lm.left_ankle, ff.left_knee_angle)
    if lm.right_hip and lm.right_knee and lm.right_ankle and ff.right_knee_angle:
        frame = draw_angle(frame, lm.right_hip, lm.right_knee, lm.right_ankle, ff.right_knee_angle)
    if lm.left_shoulder and lm.left_hip and lm.left_knee and ff.left_hip_angle:
        frame = draw_angle(frame, lm.left_shoulder, lm.left_hip, lm.left_knee, ff.left_hip_angle)

    return frame, metrics


# ---------------------------------------------------------------------------
# Full video analysis  (POST /analyze upload endpoint)
# ---------------------------------------------------------------------------

def analyze_squat(video_path: str) -> dict:
    """
    Analyze a full squat video. Returns score, metrics, feedback, score_breakdown.
    """
    pose_estimator = PoseEstimator(running_mode=vision.RunningMode.VIDEO)
    try:
        all_keypoints = pose_estimator.extract_keypoints_from_video(video_path)
    finally:
        pose_estimator.close()

    if not all_keypoints:
        return {
            "score":    0,
            "metrics":  {},
            "feedback": ["No pose detected. Ensure your full body is visible in the video."],
        }

    # Build per-frame features using the named landmark framework
    frame_features = [
        FrameFeatures.from_landmarks(PoseLandmarks(kp))
        for kp in all_keypoints
    ]

    # Aggregate into rep-level SquatFeatures
    extractor = FeatureExtractor()
    squat_features = extractor.aggregate(frame_features)

    # Score
    breakdown = score_squat(squat_features)
    feedback  = generate_feedback(squat_features)

    return {
        "score":           int(round(breakdown.total)),
        "score_breakdown": breakdown.to_dict(),
        "metrics":         squat_features.to_dict(),
        "feedback":        feedback,
    }
