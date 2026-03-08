"""
Detect squat repetitions.

Approach:
Use hip vertical position to find peaks (bottom of squat).
"""

import numpy as np
from scipy.signal import find_peaks


def count_reps(hip_positions):

    hip_positions = np.array(hip_positions)

    # Invert signal to detect bottom
    inverted = -hip_positions

    peaks, _ = find_peaks(inverted, distance=20)

    return len(peaks), peaks.tolist()