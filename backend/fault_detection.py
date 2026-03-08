"""
Detect common squat faults using pose keypoints.

Faults detected:
- Knee valgus
- Heel lift
- Excessive forward lean
- Left/right asymmetry
"""

import numpy as np


def detect_knee_valgus(left_hip, left_knee, left_ankle,
                      right_hip, right_knee, right_ankle):

    """
    Knee valgus occurs when knees collapse inward
    relative to hip and ankle alignment.
    """

    left_line = np.array(left_ankle) - np.array(left_hip)
    left_knee_offset = np.array(left_knee) - np.array(left_hip)

    right_line = np.array(right_ankle) - np.array(right_hip)
    right_knee_offset = np.array(right_knee) - np.array(right_hip)

    left_deviation = np.linalg.norm(left_knee_offset - left_line)
    right_deviation = np.linalg.norm(right_knee_offset - right_line)

    return bool(left_deviation > 0.08 or right_deviation > 0.08)


def detect_heel_lift(ankle_positions):

    """
    Heel lift detected if ankle vertical motion
    increases significantly during squat bottom.
    """

    ankle_positions = np.array(ankle_positions)

    motion = np.max(ankle_positions) - np.min(ankle_positions)

    return bool(motion > 0.05)


def detect_asymmetry(left_knee_angles, right_knee_angles):

    """
    Compare left/right knee flexion.
    """

    left_min = min(left_knee_angles)
    right_min = min(right_knee_angles)

    difference = abs(left_min - right_min)

    return bool(difference > 10)


def detect_forward_lean(back_angles):

    """
    Excessive forward lean when torso
    exceeds safe angle threshold.
    """

    return bool(max(back_angles) > 50)