"""
Visualization utilities for overlaying pose landmarks and angles on frames.
"""

import cv2
import numpy as np


# MediaPipe pose connections (pairs of landmark indices)
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # arms
    (11, 23), (12, 24), (23, 24),                        # torso
    (23, 25), (25, 27), (27, 29), (27, 31),              # left leg
    (24, 26), (26, 28), (28, 30), (28, 32),              # right leg
]


def _to_pixel(landmark, frame):
    """Convert normalized (x, y) to pixel coords."""
    h, w = frame.shape[:2]
    return int(landmark[0] * w), int(landmark[1] * h)


def draw_landmarks(frame: np.ndarray, keypoints: list) -> np.ndarray:
    """
    Draw landmark dots and skeleton connections on frame.

    Args:
        frame: BGR image.
        keypoints: List of (x, y, z, visibility) normalized landmarks.

    Returns:
        Annotated frame.
    """
    if keypoints is None:
        return frame

    # Draw connections
    for a, b in POSE_CONNECTIONS:
        if a < len(keypoints) and b < len(keypoints):
            if keypoints[a][3] > 0.5 and keypoints[b][3] > 0.5:
                pt1 = _to_pixel(keypoints[a], frame)
                pt2 = _to_pixel(keypoints[b], frame)
                cv2.line(frame, pt1, pt2, (0, 200, 255), 2)

    # Draw dots
    for lm in keypoints:
        if lm[3] > 0.5:
            px, py = _to_pixel(lm, frame)
            cv2.circle(frame, (px, py), 5, (0, 255, 0), -1)
            cv2.circle(frame, (px, py), 5, (255, 255, 255), 1)

    return frame


def draw_angle(frame: np.ndarray, a, b, c, angle: float) -> np.ndarray:
    """
    Draw the angle arc and label at joint b.

    Args:
        frame: BGR image.
        a, b, c: Landmarks (x, y, z, visibility) defining the angle.
                 b is the vertex joint.
        angle: Angle in degrees to display.

    Returns:
        Annotated frame.
    """
    if None in (a, b, c):
        return frame

    px, py = _to_pixel(b, frame)

    # Choose color: green=good, yellow=caution, red=bad (for joint angles)
    if angle < 90:
        color = (0, 200, 80)
    elif angle < 120:
        color = (0, 200, 255)
    else:
        color = (0, 100, 255)

    # Draw arc suggestion
    cv2.circle(frame, (px, py), 18, color, 2)

    # Draw angle text with background
    label = f"{int(angle)}\u00b0"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    offset_x = px + 22
    offset_y = py + 6

    # Background box
    cv2.rectangle(
        frame,
        (offset_x - 2, offset_y - th - 2),
        (offset_x + tw + 2, offset_y + 2),
        (20, 20, 20),
        -1,
    )
    cv2.putText(
        frame,
        label,
        (offset_x, offset_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
    )

    return frame
