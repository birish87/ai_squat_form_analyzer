"""
Detect squat phases using hip vertical motion.
"""

import numpy as np


def detect_phases(hip_positions):

    hip_positions = np.array(hip_positions)

    bottom = np.argmax(hip_positions)

    return {
        "descent_frames": bottom,
        "ascent_frames": len(hip_positions) - bottom
    }