"""
3D joint angle calculations.
"""

import numpy as np


def calculate_angle_3d(a, b, c):

    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine = np.dot(ba, bc) / (
        np.linalg.norm(ba) * np.linalg.norm(bc)
    )

    angle = np.arccos(np.clip(cosine, -1.0, 1.0))

    return np.degrees(angle)


def knee_angle(hip, knee, ankle):

    return calculate_angle_3d(hip, knee, ankle)


def hip_angle(shoulder, hip, knee):

    return calculate_angle_3d(shoulder, hip, knee)


def back_angle(shoulder, hip):

    vertical = np.array([0, 1, 0])
    torso = np.array(shoulder) - np.array(hip)

    cosine = np.dot(torso, vertical) / (
        np.linalg.norm(torso) * np.linalg.norm(vertical)
    )

    return np.degrees(np.arccos(cosine))