"""
Detect squat repetitions from hip vertical position.

Approach
--------
Hip y increases downward in MediaPipe coords, so the bottom of a squat
is a local MAXIMUM in the hip_positions signal.

find_peaks parameters
---------------------
distance   : minimum frames between reps (~20 frames @ 30fps = 0.67s)
prominence : how much the peak must stand out from surrounding signal.
             Prevents noise and small weight-shifts from counting as reps.
             0.02 = 2% of normalised frame height — tuned empirically.
height     : minimum hip_y value to qualify — filters out frames where
             the person is just standing (hip high = low y value).
             Set dynamically as mean + 0.3 * std of the signal so it
             adapts to camera distance and person height.
"""

import numpy as np
from scipy.signal import find_peaks, savgol_filter


def count_reps(hip_positions):
    hip_positions = np.array(hip_positions, dtype=float)

    if len(hip_positions) < 10:
        return 0, []

    # Smooth before peak detection to remove per-frame jitter
    window = min(11, len(hip_positions) if len(hip_positions) % 2 == 1
                 else len(hip_positions) - 1)
    if window >= 5:
        smoothed = savgol_filter(hip_positions, window_length=window, polyorder=2)
    else:
        smoothed = hip_positions

    # Hip y is normalised 0-1 in MediaPipe coords (y increases downward).
    # Typical standing hip_y ~0.50, squat bottom ~0.60-0.70.
    # Total range of motion is usually 0.05-0.20 in these coords.
    #
    # height threshold: hips must be lower than standing position
    #   — set at mean + 20% of the signal's range to ignore standing frames
    sig_range = float(np.max(smoothed) - np.min(smoothed))
    height_threshold = float(np.mean(smoothed) + 0.2 * sig_range)

    # prominence: peak must stand out by at least 30% of total range
    #   — filters noise and micro-shifts without killing real reps
    prom_threshold = max(0.01, 0.3 * sig_range)

    peaks, props = find_peaks(
        smoothed,
        distance=20,                  # min ~0.67s between reps at 30fps
        prominence=prom_threshold,
        height=height_threshold,
    )

    return len(peaks), peaks.tolist()