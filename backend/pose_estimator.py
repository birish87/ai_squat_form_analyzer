"""
Pose estimation using MediaPipe PoseLandmarker.

Supports:
- Single image inference (IMAGE mode)
- Video file ingestion frame-by-frame (VIDEO mode)
- Live webcam/stream frames (VIDEO mode with timestamp)
"""

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "models/pose_landmarker.task"


class PoseEstimator:
    def __init__(self, model_path=MODEL_PATH, running_mode=vision.RunningMode.VIDEO):
        """
        Initialize the PoseLandmarker.

        Args:
            model_path: Path to the .task model file.
            running_mode: MediaPipe RunningMode (IMAGE or VIDEO).
        """
        self.running_mode = running_mode
        options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=model_path),
            running_mode=running_mode,
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)
        print("PoseEstimator initialized ✅")

    def extract_keypoints_from_frame(self, image: np.ndarray, timestamp_ms: int = 0):
        """
        Detect pose landmarks in a single BGR frame.

        Args:
            image: BGR numpy array (OpenCV format).
            timestamp_ms: Timestamp in milliseconds (required for VIDEO mode).

        Returns:
            List of (x, y, z, visibility) tuples, or None if no pose detected.
        """
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        if self.running_mode == vision.RunningMode.VIDEO:
            result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        else:
            result = self.landmarker.detect(mp_image)

        if result.pose_landmarks:
            lms = result.pose_landmarks[0]  # first detected person
            return [(lm.x, lm.y, lm.z, lm.visibility) for lm in lms]

        return None

    def extract_keypoints_from_video(self, video_path: str):
        """
        Process a full video file and return per-frame keypoints.

        Args:
            video_path: Path to input video file.

        Returns:
            List of keypoint lists, one per frame where a pose was detected.
            Each element is a list of (x, y, z, visibility) tuples.
            Frames where no pose is detected are skipped.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        all_keypoints = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp_ms = int((frame_idx / fps) * 1000)
            keypoints = self.extract_keypoints_from_frame(frame, timestamp_ms)

            if keypoints is not None:
                all_keypoints.append(keypoints)

            frame_idx += 1

        cap.release()
        print(f"Processed {frame_idx} frames, detected pose in {len(all_keypoints)} ✅")
        return all_keypoints

    def close(self):
        """Release MediaPipe resources."""
        self.landmarker.close()
