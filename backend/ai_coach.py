"""
Feedback generation from SquatFeatures.
Generates prioritised, actionable coaching cues.
Falls back gracefully to the old metrics-dict interface.
"""
from __future__ import annotations
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from feature_extractor import SquatFeatures


def generate_feedback(features_or_metrics) -> List[str]:
    if isinstance(features_or_metrics, dict):
        return _feedback_from_dict(features_or_metrics)
    return _feedback_from_features(features_or_metrics)


def _feedback_from_features(f: "SquatFeatures") -> List[str]:
    feedback = []

    if f.min_knee_angle > 115:
        feedback.append(
            f"Squat deeper — your knee angle reached {f.min_knee_angle:.0f}deg. "
            "Aim for 90deg (parallel) or below for full muscle activation."
        )
    elif f.min_knee_angle > 100:
        feedback.append(
            "You're close to parallel depth. Drive your hips back and down slightly further."
        )

    if f.fault_forward_lean:
        feedback.append(
            f"Your torso leaned forward {f.max_back_angle:.0f}deg from vertical. "
            "Brace your core, keep your chest up, and sit back into the squat."
        )

    if f.fault_knee_valgus:
        feedback.append(
            "Your knees tracked inward during the squat. "
            "Push your knees out in line with your toes and strengthen your glutes."
        )

    if f.fault_heel_lift:
        feedback.append(
            "Your heels lifted at the bottom. "
            "Work on ankle dorsiflexion mobility and keep weight over mid-foot."
        )

    if f.fault_asymmetry:
        feedback.append(
            f"Left-right knee asymmetry averaged {f.mean_knee_asymmetry:.0f}deg. "
            "Focus on equal loading through both legs."
        )

    if not feedback:
        feedback.append("Great squat — no major faults detected. Keep it up.")

    return feedback[:3]


def _feedback_from_dict(metrics: dict) -> List[str]:
    feedback = []
    faults = metrics.get("faults", {})
    if faults.get("knee_valgus"):
        feedback.append("Your knees collapse inward. Push knees out and strengthen your glutes.")
    if faults.get("heel_lift"):
        feedback.append("Your heels lift off the ground. Improve ankle mobility.")
    if faults.get("forward_lean"):
        feedback.append("Your torso leans forward. Engage your core and keep chest upright.")
    if faults.get("asymmetry"):
        feedback.append("Asymmetric squat detected. Focus on equal force through both legs.")
    if not feedback:
        feedback.append("Squat mechanics look strong with no major faults detected.")
    return feedback[:3]
