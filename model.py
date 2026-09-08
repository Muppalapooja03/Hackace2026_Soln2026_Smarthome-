"""
AI Component — Isolation Forest Anomaly Detection
--------------------------------------------------
Trains on "normal" household activity patterns and scores new events for
anomalousness. Designed to be imported by the FastAPI backend, but also
runnable standalone for a quick demo.

Feature engineering turns each raw event into a numeric vector:
  - hour (0-23)
  - day_of_week (0-6)
  - is_weekend (0/1)
  - device_id (one-hot encoded)
  - event_type (one-hot encoded)
  - rolling event count in the last 10 minutes (burst detection)

Device/event-type are one-hot rather than ordinal encoded on purpose:
ordinal encoding assigns categories an arbitrary numeric order (e.g.
"energy_meter" < "front_door"), and IsolationForest can end up splitting
on that meaningless order, flagging perfectly normal events from
whichever category happens to land at a numeric extreme. One-hot avoids
inventing an ordering that isn't there.
"""

import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder

from simulator import DEVICE_TYPES, EVENT_TYPES, generate_events

ALL_EVENT_TYPES = sorted({et for types in EVENT_TYPES.values() for et in types})

DEVICE_ENCODER = OneHotEncoder(categories=[DEVICE_TYPES], sparse_output=False,
                                handle_unknown="ignore")
EVENT_TYPE_ENCODER = OneHotEncoder(categories=[ALL_EVENT_TYPES], sparse_output=False,
                                    handle_unknown="ignore")

DEVICE_COLUMNS = [f"device__{d}" for d in DEVICE_TYPES]
EVENT_TYPE_COLUMNS = [f"event__{e}" for e in ALL_EVENT_TYPES]

FEATURE_COLUMNS = (["hour", "day_of_week", "is_weekend", "burst_count"]
                    + DEVICE_COLUMNS + EVENT_TYPE_COLUMNS)


def _add_burst_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Count how many events happened in the 10 minutes before each event
    (same household) — a simple proxy for unusual bursts of activity."""
    df = df.sort_values("timestamp").reset_index(drop=True)
    ts = pd.to_datetime(df["timestamp"])
    burst_counts = []
    for i, t in enumerate(ts):
        window_start = t - pd.Timedelta(minutes=10)
        count = ((ts.iloc[:i] > window_start) & (ts.iloc[:i] <= t)).sum()
        burst_counts.append(count)
    df["burst_count"] = burst_counts
    return df


def featurize(events: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(events)
    if df.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS)

    device_oh = DEVICE_ENCODER.fit_transform(df[["device_id"]])
    event_oh = EVENT_TYPE_ENCODER.fit_transform(df[["event_type"]])
    df = pd.concat([
        df.reset_index(drop=True),
        pd.DataFrame(device_oh, columns=DEVICE_COLUMNS),
        pd.DataFrame(event_oh, columns=EVENT_TYPE_COLUMNS),
    ], axis=1)
    df = _add_burst_feature(df)
    return df


class AnomalyDetector:
    def __init__(self, contamination: float = 0.03, random_state: int = 42):
        self.model = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=random_state,
        )
        self.is_fitted = False

    def train(self, events: list[dict]):
        df = featurize(events)
        self.model.fit(df[FEATURE_COLUMNS])
        self.is_fitted = True
        return self

    def score(self, events: list[dict]) -> pd.DataFrame:
        """Returns the original events plus anomaly_score and is_anomaly."""
        if not self.is_fitted:
            raise RuntimeError("Model must be trained before scoring.")
        df = featurize(events)
        if df.empty:
            return df

        # decision_function: higher = more normal. Flip sign so higher = riskier.
        raw_scores = self.model.decision_function(df[FEATURE_COLUMNS])
        df["anomaly_score"] = (-raw_scores * 100).round(2)  # rescale for readability
        df["is_anomaly"] = self.model.predict(df[FEATURE_COLUMNS]) == -1
        # NOTE: risk_level is intentionally NOT computed here. This method's
        # job stops at "is this statistically unusual" — turning that into a
        # risk_level (and layering in domain rules like failed logins or
        # unknown devices) is the Risk Assessment Layer's job. See
        # risk_layer.assess_risk().
        return df

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            self.model = pickle.load(f)
        self.is_fitted = True
        return self


if __name__ == "__main__":
    from risk_layer import assess_risk

    # Train on "normal" data only (no injected anomalies)
    normal_events = generate_events(datetime(2026, 8, 1), num_days=14, anomaly_rate=0.0)
    detector = AnomalyDetector().train(normal_events)
    detector.save("anomaly_model.pkl")
    print(f"Trained on {len(normal_events)} normal events, saved to anomaly_model.pkl")

    # Score a fresh batch that DOES include injected anomalies
    test_events = generate_events(datetime(2026, 8, 15), num_days=2, anomaly_rate=0.08)
    results = detector.score(test_events)

    risk_levels, reasons = [], []
    for _, row in results.iterrows():
        assessment = assess_risk(row.to_dict(), row["anomaly_score"], row["is_anomaly"])
        risk_levels.append(assessment.risk_level)
        reasons.append("; ".join(assessment.reasons))
    results["risk_level"] = risk_levels
    results["reasons"] = reasons

    flagged = results[results["risk_level"].isin(["high", "critical"])]
    print(f"\nScored {len(results)} new events, {len(flagged)} rated high/critical risk:\n")
    print(flagged[["timestamp", "device_id", "event_type", "anomaly_score",
                    "risk_level", "reasons"]].to_string(index=False))
