"""
Evaluates smoothness of squat motion.
"""

import numpy as np


def movement_consistency(hip_positions):

    velocities = np.diff(hip_positions)

    variance = np.var(velocities)

    score = max(0, 10 - variance * 100)

    return score