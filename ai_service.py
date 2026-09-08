"""
AI service layer.

Bridges the database (needed for the "burst count" feature, which looks
at recent event history) with the standalone AnomalyDetector from model.py.

On import, tries to load a previously-trained model (anomaly_model.pkl).
If none exists, trains one on fresh synthetic "normal" data so the API
is usable immediately in a hackathon/demo setting. Swap this out for a
model trained on the household's own historical data once you have some.
"""

import os
from datetime import datetime, timedelta

from sqlalchemy.orm import Session

from db_models import Event
from model import AnomalyDetector
from risk_layer import assess_risk
from simulator import generate_events

MODEL_PATH = os.path.join(os.path.dirname(__file__), "anomaly_model.pkl")
BURST_WINDOW_MINUTES = 10

detector = AnomalyDetector()

if os.path.exists(MODEL_PATH):
    detector.load(MODEL_PATH)
else:
    _normal_events = generate_events(datetime.utcnow() - timedelta(days=14),
                                      num_days=14, anomaly_rate=0.0)
    detector.train(_normal_events)
    detector.save(MODEL_PATH)


def _event_to_dict(e: Event) -> dict:
    return {
        "event_id": e.event_uuid,
        "timestamp": e.timestamp.isoformat(),
        "device_id": e.device_id,
        "event_type": e.event_type,
        "hour": e.hour,
        "day_of_week": e.day_of_week,
        "is_weekend": e.is_weekend,
    }


def score_event(db: Session, event: Event) -> dict:
    """Score a single event, using recent DB history for the burst-count
    feature. Returns a dict with event_id, anomaly_score, is_anomaly, risk_level.
    """
    window_start = event.timestamp - timedelta(minutes=BURST_WINDOW_MINUTES)
    context_events = (
        db.query(Event)
        .filter(Event.timestamp > window_start, Event.timestamp <= event.timestamp)
        .order_by(Event.timestamp.asc())
        .all()
    )
    # Make sure the target event itself is included (it will be, from the
    # filter above, since window_start < timestamp <= timestamp) and ends up
    # last after the model's internal sort-by-timestamp.
    event_dicts = [_event_to_dict(e) for e in context_events]

    results = detector.score(event_dicts)
    if results.empty:
        raise RuntimeError("Scoring produced no results — unexpected empty context.")

    # The target event has the max timestamp in the window; after the
    # model's internal sort it's the last row. Ties (same-millisecond
    # events) are a known edge case for a hackathon MVP.
    target_row = results.iloc[-1]
    anomaly_score = float(target_row["anomaly_score"])
    is_anomaly = bool(target_row["is_anomaly"])

    # Risk Assessment Layer: combines the raw ML score with domain rules
    # (failed logins, unknown devices, late-night entry, energy spikes) —
    # see risk_layer.py for why this is a separate step from the model.
    assessment = assess_risk(_event_to_dict(event), anomaly_score, is_anomaly)

    return {
        "event_id": event.id,
        "device_id": event.device_id,
        "event_type": event.event_type,
        "anomaly_score": anomaly_score,
        "is_anomaly": is_anomaly,
        "risk_level": assessment.risk_level,
        "risk_reasons": assessment.reasons,
    }


def score_events(db: Session, events: list[Event]) -> list[dict]:
    return [score_event(db, e) for e in events]
