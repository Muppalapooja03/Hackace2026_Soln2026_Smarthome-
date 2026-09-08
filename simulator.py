"""
Simulated Smart-Home Event Generator
-------------------------------------
Generates synthetic events (door, motion, window, camera, network auth,
energy meter, unknown devices) that follow a plausible daily routine,
with a configurable fraction of injected "anomalous" events (odd hours,
failed-login bursts, unrecognized devices, energy spikes).

Each event is a flat dict, ready to be POSTed to the FastAPI /events
endpoint or fed straight into the anomaly model for offline training.
"""

import random
import uuid
from datetime import datetime, timedelta

DEVICE_TYPES = ["front_door", "back_door", "garage_door", "window_living_room",
                 "window_bedroom", "motion_hallway", "motion_kitchen", "camera_porch",
                 "network_auth", "energy_meter", "unknown_device"]

EVENT_TYPES = {
    "front_door": ["open", "close"],
    "back_door": ["open", "close"],
    "garage_door": ["open", "close"],
    "window_living_room": ["open", "close"],
    "window_bedroom": ["open", "close"],
    "motion_hallway": ["motion_detected"],
    "motion_kitchen": ["motion_detected"],
    "camera_porch": ["person_detected", "package_detected"],
    "network_auth": ["login_success", "login_failed"],
    "energy_meter": ["usage_normal", "usage_spike"],
    "unknown_device": ["device_detected"],
}

# Devices/event-types that only ever show up as part of an anomalous
# injection (never generated as part of "normal" background traffic).
ANOMALY_ONLY_EVENTS = [
    ("network_auth", "login_failed"),
    ("unknown_device", "device_detected"),
    ("energy_meter", "usage_spike"),
]

# Rough "normal" activity likelihood by hour (0-23), higher = more likely event fires
NORMAL_HOURLY_WEIGHTS = [
    0.05, 0.03, 0.02, 0.02, 0.03, 0.10,  # 0-5  (night, low)
    0.30, 0.60, 0.70, 0.40, 0.30, 0.30,  # 6-11 (morning ramp-up)
    0.35, 0.30, 0.30, 0.35, 0.45, 0.65,  # 12-17 (afternoon)
    0.75, 0.70, 0.55, 0.35, 0.20, 0.10,  # 18-23 (evening wind-down)
]


def _normal_event(ts: datetime) -> dict:
    """Pick a device/event combo for ordinary background traffic — never
    one of the anomaly-only combos (failed logins, unknown devices, energy
    spikes), so those stay meaningful signals rather than routine noise."""
    while True:
        device = random.choice(DEVICE_TYPES)
        event_type = random.choice(EVENT_TYPES[device])
        if (device, event_type) not in ANOMALY_ONLY_EVENTS:
            break
    return device, event_type, ts


def _anomalous_event(ts: datetime):
    """Produce one of a few break-in-adjacent anomaly flavors: an odd-hour
    door/window open, a failed login, an unrecognized device, or an energy
    usage spike."""
    flavor = random.choice(["odd_hour_entry", "failed_login", "unknown_device", "energy_spike"])
    odd_hour = random.choice([1, 2, 3, 4])
    ts = ts.replace(hour=odd_hour, minute=random.randint(0, 59))

    if flavor == "odd_hour_entry":
        device = random.choice(["front_door", "back_door", "garage_door",
                                 "window_living_room", "window_bedroom"])
        event_type = "open"
    elif flavor == "failed_login":
        device, event_type = "network_auth", "login_failed"
    elif flavor == "unknown_device":
        device, event_type = "unknown_device", "device_detected"
    else:  # energy_spike
        device, event_type = "energy_meter", "usage_spike"

    return device, event_type, ts


def _random_event(ts: datetime, anomalous: bool = False) -> dict:
    if anomalous:
        device, event_type, ts = _anomalous_event(ts)
    else:
        device, event_type, ts = _normal_event(ts)

    return {
        "event_id": str(uuid.uuid4()),
        "timestamp": ts.isoformat(),
        "device_id": device,
        "event_type": event_type,
        "hour": ts.hour,
        "day_of_week": ts.weekday(),  # 0=Mon
        "is_weekend": int(ts.weekday() >= 5),
        # Ground-truth label for evaluation/demo purposes only — strip this
        # before sending to the API, since a real device would never know
        # whether its own event "is" an anomaly.
        "_injected_anomaly": anomalous,
    }


def generate_events(start: datetime, num_days: int = 7,
                     anomaly_rate: float = 0.03, seed: int | None = 42) -> list[dict]:
    """Generate `num_days` worth of synthetic events starting at `start`.

    anomaly_rate: fraction of generated events that are deliberately anomalous
    (useful for testing detection, NOT for training the "normal" model).
    """
    if seed is not None:
        random.seed(seed)

    events = []
    current = start
    end = start + timedelta(days=num_days)

    while current < end:
        hour_weight = NORMAL_HOURLY_WEIGHTS[current.hour]
        if random.random() < hour_weight:
            is_anomaly = random.random() < anomaly_rate
            events.append(_random_event(current, anomalous=is_anomaly))
        current += timedelta(minutes=5)

    return events


if __name__ == "__main__":
    sample = generate_events(datetime(2026, 8, 1), num_days=3)
    print(f"Generated {len(sample)} events")
    for e in sample[:5]:
        print(e)
