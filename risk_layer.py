"""
Risk Assessment Layer
----------------------
Sits between the AI Anomaly Engine (Isolation Forest — model.py) and the
alerting/dashboard layer. Its job: "is this anomaly a real threat or just
noise?"

The Isolation Forest score alone is a decent general-purpose novelty
signal, but it's diluted across many features and can miss or bury
specific known-risky patterns (a failed login, a device nobody
recognizes, power draw that doesn't match anything). This layer adds
explicit domain rules on top of the raw ML score, so a handful of
well-understood red flags always get surfaced clearly, while everything
else still falls back on the general anomaly score.

This is a deliberate design choice, not a workaround: pure unsupervised
anomaly detection struggles to distinguish "genuinely rare but harmless"
from "rare and dangerous" — that judgment call is exactly what a rules
layer is for.
"""

from dataclasses import dataclass

# Ordered so we can escalate by moving forward, never backward
RISK_ORDER = ["low", "medium", "high", "critical"]

# Score buckets for the base (ML-only) risk level. Tuned against this
# feature set's typical decision_function range — retune if you change
# the features in model.py.
BASE_SCORE_BINS = [
    (-float("inf"), "low"),
    (-2.0, "medium"),
    (0.0, "high"),
    (1.5, "critical"),
]


@dataclass
class RiskAssessment:
    risk_level: str
    base_risk_level: str
    escalated: bool
    reasons: list[str]


def _base_risk_from_score(anomaly_score: float) -> str:
    level = "low"
    for threshold, label in BASE_SCORE_BINS:
        if anomaly_score >= threshold:
            level = label
    return level


def _escalate(level: str, steps: int = 1) -> str:
    idx = min(RISK_ORDER.index(level) + steps, len(RISK_ORDER) - 1)
    return RISK_ORDER[idx]


def assess_risk(event: dict, anomaly_score: float, is_anomaly: bool) -> RiskAssessment:
    """event needs at least: device_id, event_type, hour.
    Domain rules can escalate (never downgrade) the ML-derived base risk.
    """
    base = _base_risk_from_score(anomaly_score)
    level = base
    reasons: list[str] = []

    device = event.get("device_id")
    event_type = event.get("event_type")
    hour = event.get("hour")

    if device == "network_auth" and event_type == "login_failed":
        level = _escalate(level, 2)
        reasons.append("Failed login attempt")

    if device == "unknown_device":
        level = _escalate(level, 2)
        reasons.append("Unrecognized device on network")

    if device == "energy_meter" and event_type == "usage_spike":
        level = _escalate(level, 2)
        reasons.append("Unusual energy usage spike")

    if (event_type == "open"
            and device in ("front_door", "back_door", "garage_door",
                            "window_living_room", "window_bedroom")
            and hour is not None and hour in (0, 1, 2, 3, 4, 5)):
        level = _escalate(level, 2)
        reasons.append("Entry point opened during late-night hours")

    if is_anomaly and not reasons:
        reasons.append("Flagged as statistically unusual by the anomaly model")
    elif not reasons:
        reasons.append("No anomaly indicators")

    return RiskAssessment(
        risk_level=level,
        base_risk_level=base,
        escalated=(level != base),
        reasons=reasons,
    )
