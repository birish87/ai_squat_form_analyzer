"""
Pose-based feature extraction framework.

Architecture
------------
Raw keypoints (list of 33 landmarks from MediaPipe)
    → PoseLandmarks   — named access, no magic numbers, visibility gating
    → FrameFeatures   — all biomechanical values for one frame
    → SquatFeatures   — temporally aggregated rep-level features
    → (scorer)        — receives SquatFeatures, returns score + feedback

This layered design makes it trivial to extend to new drills (volleyball jump,
approach, arm-swing) by subclassing FrameFeatures or adding a new aggregator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np

from biomechanics import calculate_angle_3d, back_angle as _back_angle
from smoothing import smooth_signal

# ---------------------------------------------------------------------------
# MediaPipe landmark index constants — single source of truth
# ---------------------------------------------------------------------------

class MP:
    """MediaPipe PoseLandmarker indices."""
    NOSE            = 0
    LEFT_EYE        = 1;  RIGHT_EYE        = 2
    LEFT_EAR        = 3;  RIGHT_EAR        = 4
    LEFT_SHOULDER   = 11; RIGHT_SHOULDER   = 12
    LEFT_ELBOW      = 13; RIGHT_ELBOW      = 14
    LEFT_WRIST      = 15; RIGHT_WRIST      = 16
    LEFT_HIP        = 23; RIGHT_HIP        = 24
    LEFT_KNEE       = 25; RIGHT_KNEE       = 26
    LEFT_ANKLE      = 27; RIGHT_ANKLE      = 28
    LEFT_HEEL       = 29; RIGHT_HEEL       = 30
    LEFT_FOOT_INDEX = 31; RIGHT_FOOT_INDEX = 32

    # Convenience groups
    LOWER_BODY = [LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE]
    UPPER_BODY = [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW]


# ---------------------------------------------------------------------------
# PoseLandmarks — named, visibility-gated landmark access
# ---------------------------------------------------------------------------

Vec3 = Tuple[float, float, float]
VISIBILITY_THRESHOLD = 0.5


class PoseLandmarks:
    """
    Wraps a single frame's keypoint list with named accessors.

    Each accessor returns an (x, y, z) tuple or None if the landmark
    is below the visibility threshold — callers must handle None.

    This eliminates all magic-number indexing from downstream code.
    """

    def __init__(self, keypoints: list):
        # keypoints: list of (x, y, z, visibility) from MediaPipe
        self._kp = keypoints

    def _get(self, idx: int) -> Optional[Vec3]:
        lm = self._kp[idx]
        if lm[3] < VISIBILITY_THRESHOLD:
            return None
        return (lm[0], lm[1], lm[2])

    def _get_raw(self, idx: int) -> Vec3:
        """Return xyz regardless of visibility (for drawing)."""
        lm = self._kp[idx]
        return (lm[0], lm[1], lm[2])

    def visibility(self, idx: int) -> float:
        return self._kp[idx][3]

    # ── Named accessors ──────────────────────────────────────────────────

    @property
    def left_shoulder(self)  -> Optional[Vec3]: return self._get(MP.LEFT_SHOULDER)
    @property
    def right_shoulder(self) -> Optional[Vec3]: return self._get(MP.RIGHT_SHOULDER)
    @property
    def left_hip(self)       -> Optional[Vec3]: return self._get(MP.LEFT_HIP)
    @property
    def right_hip(self)      -> Optional[Vec3]: return self._get(MP.RIGHT_HIP)
    @property
    def left_knee(self)      -> Optional[Vec3]: return self._get(MP.LEFT_KNEE)
    @property
    def right_knee(self)     -> Optional[Vec3]: return self._get(MP.RIGHT_KNEE)
    @property
    def left_ankle(self)     -> Optional[Vec3]: return self._get(MP.LEFT_ANKLE)
    @property
    def right_ankle(self)    -> Optional[Vec3]: return self._get(MP.RIGHT_ANKLE)
    @property
    def left_heel(self)      -> Optional[Vec3]: return self._get(MP.LEFT_HEEL)
    @property
    def right_heel(self)     -> Optional[Vec3]: return self._get(MP.RIGHT_HEEL)
    @property
    def left_foot(self)      -> Optional[Vec3]: return self._get(MP.LEFT_FOOT_INDEX)
    @property
    def right_foot(self)     -> Optional[Vec3]: return self._get(MP.RIGHT_FOOT_INDEX)
    @property
    def nose(self)           -> Optional[Vec3]: return self._get(MP.NOSE)

    # ── Derived helpers ──────────────────────────────────────────────────

    @property
    def mid_hip(self) -> Optional[Vec3]:
        """Midpoint between left and right hip."""
        l, r = self.left_hip, self.right_hip
        if l is None or r is None:
            return None
        return ((l[0]+r[0])/2, (l[1]+r[1])/2, (l[2]+r[2])/2)

    @property
    def mid_shoulder(self) -> Optional[Vec3]:
        l, r = self.left_shoulder, self.right_shoulder
        if l is None or r is None:
            return None
        return ((l[0]+r[0])/2, (l[1]+r[1])/2, (l[2]+r[2])/2)

    def lower_body_visible(self) -> bool:
        """True if all key lower-body landmarks are above visibility threshold."""
        return all(self._get(i) is not None for i in MP.LOWER_BODY)

    def raw_keypoints(self) -> list:
        return self._kp


# ---------------------------------------------------------------------------
# FrameFeatures — all biomechanical values for one frame
# ---------------------------------------------------------------------------

@dataclass
class FrameFeatures:
    """
    Every biomechanical measurement extracted from a single frame.

    None values indicate the landmark was not visible enough to compute
    the angle reliably — downstream code must handle gracefully.
    """
    # Joint angles (degrees)
    left_knee_angle:  Optional[float] = None
    right_knee_angle: Optional[float] = None
    left_hip_angle:   Optional[float] = None
    right_hip_angle:  Optional[float] = None
    back_angle:       Optional[float] = None   # torso vs vertical
    ankle_dorsiflexion_left:  Optional[float] = None
    ankle_dorsiflexion_right: Optional[float] = None

    # Positional features (normalized 0-1)
    hip_height:       Optional[float] = None   # mid-hip y (higher y = lower body)
    left_ankle_y:     Optional[float] = None
    right_ankle_y:    Optional[float] = None

    # Symmetry
    knee_angle_diff:  Optional[float] = None   # |left - right| knee angle

    # Lateral alignment (for valgus detection)
    # Positive = knee inside hip-ankle line (valgus), negative = outside (varus)
    left_knee_valgus_ratio:  Optional[float] = None
    right_knee_valgus_ratio: Optional[float] = None

    # Landmarks (for visualization)
    landmarks: Optional[PoseLandmarks] = field(default=None, repr=False)

    @classmethod
    def from_landmarks(cls, lm: PoseLandmarks) -> "FrameFeatures":
        """
        Compute all features from a PoseLandmarks instance.
        Returns a FrameFeatures with None for any uncomputable value.
        """
        f = cls(landmarks=lm)

        if not lm.lower_body_visible():
            return f  # can't compute anything useful

        lh  = lm.left_hip;    rh  = lm.right_hip
        lk  = lm.left_knee;   rk  = lm.right_knee
        la  = lm.left_ankle;  ra  = lm.right_ankle
        ls  = lm.left_shoulder; rs = lm.right_shoulder

        # ── Knee angles ──────────────────────────────────────────────
        if lh and lk and la:
            f.left_knee_angle = float(calculate_angle_3d(lh, lk, la))
        if rh and rk and ra:
            f.right_knee_angle = float(calculate_angle_3d(rh, rk, ra))

        if f.left_knee_angle is not None and f.right_knee_angle is not None:
            f.knee_angle_diff = abs(f.left_knee_angle - f.right_knee_angle)

        # ── Hip angles ───────────────────────────────────────────────
        if ls and lh and lk:
            f.left_hip_angle = float(calculate_angle_3d(ls, lh, lk))
        if rs and rh and rk:
            f.right_hip_angle = float(calculate_angle_3d(rs, rh, rk))

        # ── Back angle (torso lean from vertical) ────────────────────
        # Use midpoints for robustness against single-side occlusion
        ms = lm.mid_shoulder; mh = lm.mid_hip
        if ms and mh:
            f.back_angle = float(_back_angle(ms, mh))
        elif ls and lh:
            f.back_angle = float(_back_angle(ls, lh))
        elif rs and rh:
            f.back_angle = float(_back_angle(rs, rh))

        # ── Ankle dorsiflexion (shin angle from vertical) ────────────
        if lk and la:
            f.ankle_dorsiflexion_left = float(_back_angle(lk, la))
        if rk and ra:
            f.ankle_dorsiflexion_right = float(_back_angle(rk, ra))

        # ── Positional ───────────────────────────────────────────────
        mh = lm.mid_hip
        if mh:
            f.hip_height = float(mh[1])   # y increases downward in MediaPipe
        if la:
            f.left_ankle_y  = float(la[1])
        if ra:
            f.right_ankle_y = float(ra[1])

        # ── Lateral knee tracking (valgus detection) ─────────────────
        # Project the knee onto the hip-ankle line and measure deviation.
        # We interpolate where the hip-ankle line passes at the knee's y-height,
        # then compare the knee's actual x to that expected x.
        # Normalise by hip-ankle distance so the result is camera-scale independent.
        # Positive ratio = knee is medial to the line (valgus).
        if lh and lk and la:
            hip_ankle_dy = la[1] - lh[1]
            if abs(hip_ankle_dy) > 1e-4:
                t = (lk[1] - lh[1]) / hip_ankle_dy   # 0=hip, 1=ankle
                expected_x = lh[0] + t * (la[0] - lh[0])
                hip_ankle_dist = ((la[0]-lh[0])**2 + (la[1]-lh[1])**2) ** 0.5
                if hip_ankle_dist > 1e-4:
                    f.left_knee_valgus_ratio = float((expected_x - lk[0]) / hip_ankle_dist)
        if rh and rk and ra:
            hip_ankle_dy = ra[1] - rh[1]
            if abs(hip_ankle_dy) > 1e-4:
                t = (rk[1] - rh[1]) / hip_ankle_dy
                expected_x = rh[0] + t * (ra[0] - rh[0])
                hip_ankle_dist = ((ra[0]-rh[0])**2 + (ra[1]-rh[1])**2) ** 0.5
                if hip_ankle_dist > 1e-4:
                    # Right leg: valgus means knee deviates left (negative x in MP coords)
                    f.right_knee_valgus_ratio = float((rk[0] - expected_x) / hip_ankle_dist)

        return f

    def mean_knee_angle(self) -> Optional[float]:
        vals = [v for v in [self.left_knee_angle, self.right_knee_angle] if v is not None]
        return float(np.mean(vals)) if vals else None

    def mean_hip_angle(self) -> Optional[float]:
        vals = [v for v in [self.left_hip_angle, self.right_hip_angle] if v is not None]
        return float(np.mean(vals)) if vals else None


# ---------------------------------------------------------------------------
# SquatFeatures — temporally aggregated rep-level feature set
# ---------------------------------------------------------------------------

@dataclass
class SquatFeatures:
    """
    Aggregated features across all frames of a squat rep (or full video).

    This is the input contract for the scorer — both heuristic and ML.
    All values are plain Python floats so they're directly JSON-serialisable.
    """
    # Depth
    min_knee_angle:  float = 180.0   # smaller = deeper squat
    min_hip_angle:   float = 180.0
    squat_depth_label: str = "above parallel"

    # Posture
    mean_back_angle: float = 0.0
    max_back_angle:  float = 0.0

    # Ankle mobility
    mean_dorsiflexion: float = 0.0

    # Symmetry
    mean_knee_asymmetry: float = 0.0  # mean |L-R| across frames
    max_knee_asymmetry:  float = 0.0

    # Lateral tracking
    max_left_valgus:  float = 0.0    # max inward knee deviation
    max_right_valgus: float = 0.0

    # Heel lift
    left_ankle_y_range:  float = 0.0  # large = heel lifts
    right_ankle_y_range: float = 0.0

    # Temporal
    movement_consistency: float = 10.0
    rep_count: int = 0

    # Phase timing (frames)
    descent_frames: int = 0
    ascent_frames:  int = 0

    # Fault flags (derived from above)
    fault_knee_valgus:  bool = False
    fault_heel_lift:    bool = False
    fault_asymmetry:    bool = False
    fault_forward_lean: bool = False

    def to_ml_vector(self) -> List[float]:
        """
        Fixed-length feature vector for ML model input.
        Order must match SquatNet input layer.
        """
        return [
            self.min_knee_angle / 180.0,    # normalised 0-1
            self.max_back_angle / 90.0,
            self.mean_knee_asymmetry / 30.0,
            self.left_ankle_y_range / 0.1,
        ]

    def to_dict(self) -> dict:
        return {
            "min_knee_angle":       round(self.min_knee_angle, 1),
            "min_hip_angle":        round(self.min_hip_angle, 1),
            "squat_depth":          self.squat_depth_label,
            "max_back_angle":       round(self.max_back_angle, 1),
            "mean_back_angle":      round(self.mean_back_angle, 1),
            "mean_dorsiflexion":    round(self.mean_dorsiflexion, 1),
            "mean_knee_asymmetry":  round(self.mean_knee_asymmetry, 1),
            "left_ankle_y_range":   round(self.left_ankle_y_range, 4),
            "movement_consistency": round(self.movement_consistency, 2),
            "rep_count":            self.rep_count,
            "descent_frames":       self.descent_frames,
            "ascent_frames":        self.ascent_frames,
            "faults": {
                "knee_valgus":  self.fault_knee_valgus,
                "heel_lift":    self.fault_heel_lift,
                "asymmetry":    self.fault_asymmetry,
                "forward_lean": self.fault_forward_lean,
            },
        }


# ---------------------------------------------------------------------------
# FeatureExtractor — converts a list of FrameFeatures → SquatFeatures
# ---------------------------------------------------------------------------

class FeatureExtractor:
    """
    Aggregates per-frame features into a rep-level SquatFeatures.

    Usage:
        extractor = FeatureExtractor()
        frame_features = [FrameFeatures.from_landmarks(PoseLandmarks(kp))
                          for kp in all_keypoints]
        squat_features = extractor.aggregate(frame_features)
    """

    # Fault thresholds — centralised here, easy to tune
    VALGUS_THRESHOLD    = 0.15   # ratio of hip-ankle distance; 0.15 ≈ clear inward collapse
    HEEL_LIFT_THRESHOLD = 0.025  # normalised y range
    ASYMMETRY_THRESHOLD = 12.0   # degrees L-R difference
    FORWARD_LEAN_THRESHOLD = 45.0  # degrees from vertical

    def aggregate(self, frames: List[FrameFeatures]) -> SquatFeatures:
        """Aggregate a sequence of FrameFeatures into one SquatFeatures."""
        if not frames:
            return SquatFeatures()

        sf = SquatFeatures()

        # ── Collect valid per-frame series ───────────────────────────
        knee_angles   = [f.mean_knee_angle()  for f in frames if f.mean_knee_angle()  is not None]
        hip_angles    = [f.mean_hip_angle()   for f in frames if f.mean_hip_angle()   is not None]
        back_angles   = [f.back_angle         for f in frames if f.back_angle         is not None]
        dorsiflexion  = [f.ankle_dorsiflexion_left for f in frames if f.ankle_dorsiflexion_left is not None]
        asymmetry     = [f.knee_angle_diff    for f in frames if f.knee_angle_diff    is not None]
        hip_heights   = [f.hip_height         for f in frames if f.hip_height         is not None]
        left_ankle_ys = [f.left_ankle_y       for f in frames if f.left_ankle_y       is not None]
        right_ankle_ys= [f.right_ankle_y      for f in frames if f.right_ankle_y      is not None]
        l_valgus      = [f.left_knee_valgus_ratio  for f in frames if f.left_knee_valgus_ratio  is not None]
        r_valgus      = [f.right_knee_valgus_ratio for f in frames if f.right_knee_valgus_ratio is not None]

        # ── Smooth temporal signals ───────────────────────────────────
        if len(knee_angles) >= 7:
            knee_angles = list(smooth_signal(knee_angles))
        if len(back_angles) >= 7:
            back_angles = list(smooth_signal(back_angles))
        if len(hip_heights) >= 7:
            hip_heights = list(smooth_signal(hip_heights))

        # ── Depth ────────────────────────────────────────────────────
        if knee_angles:
            sf.min_knee_angle = float(min(knee_angles))
            if sf.min_knee_angle <= 90:
                sf.squat_depth_label = "below parallel"
            elif sf.min_knee_angle <= 100:
                sf.squat_depth_label = "parallel"
            else:
                sf.squat_depth_label = "above parallel"

        if hip_angles:
            sf.min_hip_angle = float(min(hip_angles))

        # ── Posture ──────────────────────────────────────────────────
        if back_angles:
            sf.mean_back_angle = float(np.mean(back_angles))
            sf.max_back_angle  = float(max(back_angles))

        # ── Ankle mobility ───────────────────────────────────────────
        if dorsiflexion:
            sf.mean_dorsiflexion = float(np.mean(dorsiflexion))

        # ── Symmetry ─────────────────────────────────────────────────
        if asymmetry:
            sf.mean_knee_asymmetry = float(np.mean(asymmetry))
            sf.max_knee_asymmetry  = float(max(asymmetry))

        # ── Heel lift ────────────────────────────────────────────────
        if left_ankle_ys:
            sf.left_ankle_y_range  = float(max(left_ankle_ys) - min(left_ankle_ys))
        if right_ankle_ys:
            sf.right_ankle_y_range = float(max(right_ankle_ys) - min(right_ankle_ys))

        # ── Lateral valgus ───────────────────────────────────────────
        if l_valgus:
            # Only count positive values (inward deviation = valgus)
            sf.max_left_valgus  = float(max(max(v for v in l_valgus), 0.0))
        if r_valgus:
            sf.max_right_valgus = float(max(max(v for v in r_valgus), 0.0))

        # ── Movement consistency ─────────────────────────────────────
        if len(hip_heights) > 1:
            velocities = np.diff(hip_heights)
            variance   = float(np.var(velocities))
            sf.movement_consistency = max(0.0, 10.0 - variance * 100)

        # ── Phase timing ─────────────────────────────────────────────
        if hip_heights:
            bottom_idx = int(np.argmax(hip_heights))  # y max = lowest body position
            sf.descent_frames = bottom_idx
            sf.ascent_frames  = len(hip_heights) - bottom_idx

        # ── Rep count ────────────────────────────────────────────────
        from rep_counter import count_reps
        if hip_heights:
            sf.rep_count, _ = count_reps(hip_heights)

        # ── Fault flags ──────────────────────────────────────────────
        sf.fault_knee_valgus  = (sf.max_left_valgus  > self.VALGUS_THRESHOLD or
                                 sf.max_right_valgus > self.VALGUS_THRESHOLD)
        sf.fault_heel_lift    = (sf.left_ankle_y_range  > self.HEEL_LIFT_THRESHOLD or
                                 sf.right_ankle_y_range > self.HEEL_LIFT_THRESHOLD)
        sf.fault_asymmetry    = sf.max_knee_asymmetry > self.ASYMMETRY_THRESHOLD
        sf.fault_forward_lean = sf.max_back_angle > self.FORWARD_LEAN_THRESHOLD

        return sf
