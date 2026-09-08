# AI-Powered Smart Home Security System

**HackACE 2026 — Final Round · Domain 5: AI-Driven IT Solutions**
**Team:** Soln2026 · **SPOC:** Muppala Pooja ([muppalapooja03@gmail.com](mailto:muppalapooja03@gmail.com))

An AI-driven smart-home security platform that learns a household's normal activity patterns and flags genuinely risky behaviour — instead of firing the same alert for every sensor trigger. Round 2 extends the working software MVP with real ESP32-based hardware sensor nodes.

---

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [AI Implementation — Algorithm & Logic](#ai-implementation--algorithm--logic)
- [Hardware Components (Round 2)](#hardware-components-round-2)
- [Hardware Integration Architecture](#hardware-integration-architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [API Reference](#api-reference)
- [Results](#results)
- [Roadmap](#roadmap)
- [Team](#team)

---

## Overview

Traditional smart-home security systems rely on predefined rules and isolated sensor triggers — every event gets the same alert regardless of context. This causes excessive false alarms for normal activity while genuine threats that don't match a predefined rule go unnoticed.

This project implements an end-to-end pipeline that:

1. Simulates / ingests events from doors, windows, motion sensors, network logins, energy meters, and unknown devices.
2. Scores each event with an **Isolation Forest** anomaly-detection model trained on normal household activity.
3. Combines that ML score with an explicit **Risk Assessment Layer** so known-dangerous patterns (failed logins, unrecognized devices, late-night entries, energy spikes) are never missed regardless of the raw model score.
4. Surfaces everything on a live **Streamlit** dashboard.

For Round 2, the same pipeline is fed by real **ESP32-based hardware sensor nodes**, turning the simulated architecture into a working physical prototype.

## Problem Statement

- **No context** — a door sensor fires the same alert for a routine 3pm open and an unexplained 3am one.
- **No correlation** — a failed login and an unfamiliar device appearing minutes apart look unrelated to a rule-based system.
- **Alert fatigue** — event volume grows faster than any homeowner can manually review as more devices connect.

**Result:** either alert fatigue from false positives, or blind spots where genuine multi-signal threats go unflagged.

## Key Features

- Multi-device event simulation: doors/windows, motion, camera, network auth, energy meter, unknown-device detection
- Unsupervised anomaly detection — no hand-labeled attack data required
- Two-layer risk assessment (ML score + domain rules) for both adaptability and explainability
- 10-minute burst-window feature — recent activity context, not a single event in isolation
- REST API with bulk ingestion, batch analysis, and an alerts feed with an acknowledge workflow
- Live dashboard: pipeline status, key metrics, risk distribution, anomaly-score timeline, color-coded alerts
- **Round 2:** real ESP32 hardware sensor nodes feeding live events into the same pipeline

## System Architecture

```
 Events → FastAPI → Feature Engineering → Isolation Forest → Risk Layer → PostgreSQL → Dashboard
```

1. A device (or simulator) emits a raw event: device ID, event type, timestamp.
2. FastAPI validates and stores it, stamping hour / day-of-week / weekend metadata.
3. Analysis pulls each event's 10-minute activity window and engineers its feature vector.
4. The Isolation Forest returns an anomaly score and a statistical label.
5. The Risk Layer combines that score with domain rules into a final risk level.
6. High/critical results write to an alerts table.
7. The dashboard polls the API and renders live metrics, charts, and alerts.

## AI Implementation — Algorithm & Logic

**1. Feature Engineering**
Categorical fields (`device_id`, `event_type`) are one-hot encoded — an early version used ordinal encoding, which let the model split on a meaningless numeric order and flag normal energy-meter readings as high risk. A 10-minute rolling burst-window (event counts, hour, weekday/weekend) is added so scoring reflects context, not a single isolated event.

**2. Isolation Forest (unsupervised)**
Builds random partition trees over the feature vectors. Anomalies isolate in fewer splits than normal points, so path length converts directly into an anomaly score in `[0, 1]`. No labeled attack data is required — the model learns what's "normal" purely from observed activity.

**3. Risk Assessment Layer (escalation logic)**

```
final_risk = max(ml_anomaly_score, rule_escalation_score)
```

Specific domain rules — a failed login, an unrecognized device, a late-night entry-point open, or an abnormal energy spike — force an automatic escalation regardless of the raw ML score, so the system's most safety-critical judgements never depend solely on an opaque model.

**4. Why the combination wins**
The raw ML score alone caught only **11%** of injected anomalies in testing — rare signals like a failed login get diluted across many other features. Adding the rule-based Risk Layer raised recall to **100%** across two independently-seeded test batches, at **56–90%** precision.

## Hardware Components 

Compact ESP32 edge nodes cover every event type the software already models:

| Component | Role | Why | How |
|---|---|---|---|
| **ESP32** | Wi-Fi + BLE gateway MCU | Native Wi-Fi, enough compute headroom, cheap enough to place at every sensor point | Reads sensors over GPIO/UART, batches to JSON, pushes to the FastAPI `/events` endpoint |
| **Magnetic reed switch** | Door / window sensor | Simplest, most reliable open/close detection — no false triggers from light or pets | Wired to an ESP32 GPIO interrupt; a state change fires the event instantly |
| **PIR / mmWave radar** | Motion / presence sensor | Presence detection without a camera; mmWave avoids PIR's false triggers from pets/curtains | Digital output toggles on motion; ESP32 debounces and timestamps before sending |
| **PZEM-004T** | Energy meter | Real per-circuit voltage/current lets the Risk Layer catch the same energy-spike pattern it was designed for | UART/Modbus link to an ESP32, polled every few seconds |
| **ESP32-CAM** | Camera module | Visual confirmation on a high-risk alert without a separate camera system | Captures a still on trigger, uploads it with the event payload |
| **ESP32 (promiscuous mode)** | Network sniffer | Directly implements the "unrecognized device" signal the software already scores | Radio in promiscuous mode logs MAC addresses; unrecognized ones are sent as events |

## Hardware Integration Architecture

```
Reed Switch ─┐
PIR/mmWave ──┼──► ESP32 Edge Node (Wi-Fi / MQTT) ──► FastAPI Ingestion Endpoint ──► (existing software pipeline)
PZEM-004T ───┤
ESP32-CAM ───┘
```

Nothing downstream of the FastAPI ingestion endpoint changes — the same Feature Engineering → Isolation Forest → Risk Layer → PostgreSQL → Dashboard pipeline consumes hardware-sourced events exactly as it consumed simulated ones.

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.12 |
| AI / ML | scikit-learn (Isolation Forest, One-Hot Encoding), pandas, numpy |
| Backend API | FastAPI, Pydantic, Uvicorn |
| Database | PostgreSQL, SQLAlchemy ORM (psycopg2) |
| Dashboard | Streamlit, Altair |
| Hardware | ESP32, ESP32-CAM, PIR/mmWave, magnetic reed switch, PZEM-004T |

## Project Structure

> Adjust paths below to match your actual repository layout.

```
smart-home-security/
├── backend/
│   ├── main.py                # FastAPI app & routes
│   ├── models.py               # SQLAlchemy models
│   ├── schemas.py              # Pydantic request/response schemas
│   ├── feature_engineering.py  # One-hot encoding, burst-window features
│   ├── anomaly_model.py        # Isolation Forest training/inference
│   └── risk_layer.py           # Rule-based escalation logic
├── dashboard/
│   └── app.py                  # Streamlit dashboard
├── simulator/
│   └── event_simulator.py      # Synthetic multi-device event generator
├── hardware/
│   ├── esp32_gateway/           # Firmware for the main ESP32 edge node
│   ├── esp32_cam/                # Firmware for the camera node
│   └── esp32_sniffer/            # Promiscuous-mode Wi-Fi sniffer firmware
├── requirements.txt
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.12+
- PostgreSQL 14+
- Arduino IDE or PlatformIO (for ESP32 firmware)

### Backend & Dashboard

```bash
# Clone the repository
git clone <your-repo-url>
cd smart-home-security

# Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure the database connection (e.g. via a .env file)
export DATABASE_URL="postgresql://user:password@localhost:5432/smart_home_db"

# Run the FastAPI backend
uvicorn backend.main:app --reload

# In a separate terminal, run the dashboard
streamlit run dashboard/app.py
```

### Hardware Nodes

1. Open the relevant sketch under `hardware/` in the Arduino IDE / PlatformIO.
2. Set your Wi-Fi credentials and the backend's `/events` endpoint URL.
3. Flash the ESP32 board and power it near the sensor it serves (door, motion zone, meter panel, etc.).

## API Reference

> Update this section with your actual route names if they differ.

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/events` | Ingest a single event |
| `POST` | `/events/bulk` | Bulk-ingest multiple events |
| `POST` | `/analyze` | Run feature engineering + Isolation Forest + Risk Layer on pending events |
| `GET` | `/alerts` | Retrieve current alerts feed |
| `POST` | `/alerts/{id}/acknowledge` | Acknowledge an alert |
| `GET` | `/metrics` | Pipeline status and summary metrics for the dashboard |

## Results

- ML model alone: **11%** recall on injected anomalies
- ML + Risk Layer (integrated): **100%** recall across multiple test runs
- Precision with the integrated approach: **56%–90%**

These results confirm that contextual, rule-augmented risk assessment meaningfully improves detection of high-risk events over a purely statistical model.

## Roadmap

- Real device integration — ESP32 hardware nodes (delivered this round)
- Per-household model personalization
- SMS / push alerts
- Homeowner feedback loop to retrain on new normal-activity data
- Automatic model-version checking (a stale cached model previously crashed the API after a feature change)

## Team

| Hackathon ID | Name | Role | Organization |
|---|---|---|---|
| ACEIH0483 | Muppala Pooja | SPOC | Comfinity Technologies |
| ACEIH0492 | Atchaya R | Team Member | Comfinity Technologies |

---

*HackACE 2026 · Organized by KPR Institute of Engineering and Technology, Dept. of CSE & Dept. of CSE (AIML), and AllCollegeEvent.com*
