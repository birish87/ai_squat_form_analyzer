"""
Movement scoring logic.

Scoring philosophy
------------------
Each dimension of squat quality is scored independently on a 0-10 scale,
then combined via weighted average. Weights reflect coaching priorities:
depth and posture matter most; consistency and ankle mobility are bonuses.

This structure makes it trivial to retune for volleyball drill scoring
by swapping weights or adding new dimension functions.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from feature_extractor import SquatFeatures


# ---------------------------------------------------------------------------
# Dimension weights — must sum to 1.0
# ---------------------------------------------------------------------------

WEIGHTS = {
    "depth":        0.30,
    "posture":      0.25,
    "symmetry":     0.20,
    "heel_contact": 0.15,
    "consistency":  0.10,
}


@dataclass
class ScoreBreakdown:
    """Per-dimension scores for transparency and debugging."""
    depth:        float
    posture:      float
    symmetry:     float
    heel_contact: float
    consistency:  float
    total:        float

    def to_dict(self) -> dict:
        return {
            "depth":        round(self.depth, 1),
            "posture":      round(self.posture, 1),
            "symmetry":     round(self.symmetry, 1),
            "heel_contact": round(self.heel_contact, 1),
            "consistency":  round(self.consistency, 1),
            "total":        round(self.total, 1),
        }


def score_squat(features: "SquatFeatures") -> ScoreBreakdown:
    """
    Score a squat rep from a SquatFeatures object.
    Returns a ScoreBreakdown with per-dimension and total (1-10) scores.
    """
    depth        = _score_depth(features)
    posture      = _score_posture(features)
    symmetry     = _score_symmetry(features)
    heel_contact = _score_heel_contact(features)
    consistency  = _score_consistency(features)

    total = (
        depth        * WEIGHTS["depth"]        +
        posture      * WEIGHTS["posture"]       +
        symmetry     * WEIGHTS["symmetry"]      +
        heel_contact * WEIGHTS["heel_contact"]  +
        consistency  * WEIGHTS["consistency"]
    )

    total = round(max(1.0, min(10.0, total)), 1)

    return ScoreBreakdown(
        depth=depth, posture=posture, symmetry=symmetry,
        heel_contact=heel_contact, consistency=consistency, total=total,
    )


def _score_depth(f: "SquatFeatures") -> float:
    angle = f.min_knee_angle
    if angle <= 90:   return 10.0
    if angle >= 130:  return 1.0
    return 10.0 - (angle - 90) * (9.0 / 40.0)


def _score_posture(f: "SquatFeatures") -> float:
    angle = f.max_back_angle
    if angle <= 20:   return 10.0
    if angle >= 60:   return 1.0
    return 10.0 - (angle - 20) * (9.0 / 40.0)


def _score_symmetry(f: "SquatFeatures") -> float:
    diff = f.mean_knee_asymmetry
    if diff <= 2:     return 10.0
    if diff >= 25:    return 1.0
    return 10.0 - (diff - 2) * (9.0 / 23.0)


def _score_heel_contact(f: "SquatFeatures") -> float:
    lift = max(f.left_ankle_y_range, f.right_ankle_y_range)
    if lift < 0.01:   return 10.0
    if lift >= 0.05:  return 1.0
    return 10.0 - (lift - 0.01) * (9.0 / 0.04)


def _score_consistency(f: "SquatFeatures") -> float:
    return max(1.0, min(10.0, f.movement_consistency))


def heuristic_score(metrics: dict) -> int:
    """Backwards-compatible shim for raw metrics dicts."""
    score = 10
    if metrics.get("min_knee_angle", 90) > 100: score -= 3
    if metrics.get("max_back_angle", 0)  > 45:  score -= 3
    if metrics.get("min_hip_angle", 90)  > 90:  score -= 2
    return max(score, 1)
