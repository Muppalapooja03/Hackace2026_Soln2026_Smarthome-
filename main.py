"""
FastAPI backend for the smart-home security MVP.

Routes:
  POST /events              — ingest a raw event
  GET  /events               — list events (paginated, optional filters)
  POST /analyze              — run the AI model on pending (or specified) events
  GET  /alerts                — list alerts (optional filters)
  PATCH /alerts/{id}/ack      — acknowledge an alert
  GET  /health                — liveness check

Run locally:
    pip install -r requirements.txt
    uvicorn main:app --reload --port 8000

Then browse http://localhost:8000/docs for interactive Swagger UI.
"""

import uuid
from datetime import datetime
from typing import List, Optional

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

import ai_service
import schemas
from database import Base, SessionLocal, engine, get_db
from db_models import Alert, AnomalyResult, Event

# Risk levels that generate an alert (vs. just being recorded silently)
ALERT_RISK_LEVELS = {"high", "critical"}

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Smart Home Security API",
    description="Ingests smart-home events, scores them for anomalies, and surfaces alerts.",
    version="0.1.0",
)

# Wide open for hackathon demo purposes — tighten before anything real.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}


@app.post("/events", response_model=schemas.EventOut, status_code=201)
def create_event(event_in: schemas.EventIn, db: Session = Depends(get_db)):
    ts = event_in.timestamp or datetime.utcnow()

    event = Event(
        event_uuid=str(uuid.uuid4()),
        timestamp=ts,
        device_id=event_in.device_id,
        event_type=event_in.event_type,
        hour=ts.hour,
        day_of_week=ts.weekday(),
        is_weekend=int(ts.weekday() >= 5),
        analyzed=False,
    )
    db.add(event)
    db.commit()
    db.refresh(event)
    return event


@app.post("/events/bulk", response_model=List[schemas.EventOut], status_code=201)
def create_events_bulk(events_in: List[schemas.EventIn], db: Session = Depends(get_db)):
    """Ingest many events in one call — mainly for the dashboard's
    'simulate events' button, to avoid one HTTP round-trip per event."""
    created = []
    for event_in in events_in:
        ts = event_in.timestamp or datetime.utcnow()
        event = Event(
            event_uuid=str(uuid.uuid4()),
            timestamp=ts,
            device_id=event_in.device_id,
            event_type=event_in.event_type,
            hour=ts.hour,
            day_of_week=ts.weekday(),
            is_weekend=int(ts.weekday() >= 5),
            analyzed=False,
        )
        db.add(event)
        created.append(event)
    db.commit()
    for event in created:
        db.refresh(event)
    return created


@app.get("/events", response_model=List[schemas.EventOut])
def list_events(
    device_id: Optional[str] = None,
    analyzed: Optional[bool] = None,
    limit: int = Query(50, le=500),
    offset: int = 0,
    db: Session = Depends(get_db),
):
    q = db.query(Event)
    if device_id:
        q = q.filter(Event.device_id == device_id)
    if analyzed is not None:
        q = q.filter(Event.analyzed == analyzed)
    return q.order_by(Event.timestamp.desc()).offset(offset).limit(limit).all()


@app.get("/events/scored", response_model=List[schemas.EventScored])
def list_scored_events(
    risk_level: Optional[str] = None,
    limit: int = Query(200, le=1000),
    db: Session = Depends(get_db),
):
    """Events joined with their anomaly result (if analyzed) — the main
    feed the dashboard reads for its table and charts."""
    q = db.query(Event)
    if risk_level:
        q = q.join(AnomalyResult).filter(AnomalyResult.risk_level == risk_level)
    events = q.order_by(Event.timestamp.desc()).limit(limit).all()

    out = []
    for e in events:
        ar = e.anomaly_result
        out.append(schemas.EventScored(
            id=e.id, event_uuid=e.event_uuid, timestamp=e.timestamp,
            device_id=e.device_id, event_type=e.event_type, hour=e.hour,
            day_of_week=e.day_of_week, is_weekend=e.is_weekend, analyzed=e.analyzed,
            anomaly_score=ar.anomaly_score if ar else None,
            is_anomaly=ar.is_anomaly if ar else None,
            risk_level=ar.risk_level if ar else None,
        ))
    return out


@app.post("/analyze", response_model=List[schemas.AnalyzeResultOut])
def analyze(request: schemas.AnalyzeRequest, db: Session = Depends(get_db)):
    """Run the anomaly model on the given event ids, or on every
    not-yet-analyzed event if none are specified. Stores anomaly_results
    and creates an Alert for anything scored high/critical."""

    if request.event_ids:
        events = db.query(Event).filter(Event.id.in_(request.event_ids)).all()
        missing = set(request.event_ids) - {e.id for e in events}
        if missing:
            raise HTTPException(404, f"Event id(s) not found: {sorted(missing)}")
    else:
        events = db.query(Event).filter(Event.analyzed.is_(False)).all()

    if not events:
        return []

    scored = ai_service.score_events(db, events)

    results_out = []
    for event, result in zip(events, scored):
        # Upsert anomaly_result (one per event)
        existing = db.query(AnomalyResult).filter(
            AnomalyResult.event_id == event.id).first()
        if existing:
            existing.anomaly_score = result["anomaly_score"]
            existing.is_anomaly = result["is_anomaly"]
            existing.risk_level = result["risk_level"]
        else:
            db.add(AnomalyResult(
                event_id=event.id,
                anomaly_score=result["anomaly_score"],
                is_anomaly=result["is_anomaly"],
                risk_level=result["risk_level"],
            ))

        event.analyzed = True

        if result["risk_level"] in ALERT_RISK_LEVELS:
            db.add(Alert(
                event_id=event.id,
                risk_level=result["risk_level"],
                message=(f"{result['risk_level'].upper()} risk: "
                         f"{event.event_type} on {event.device_id} "
                         f"at {event.timestamp.isoformat()} "
                         f"(score {result['anomaly_score']:.2f})"),
            ))

        results_out.append(schemas.AnalyzeResultOut(**result))

    db.commit()
    return results_out


@app.get("/alerts", response_model=List[schemas.AlertOut])
def list_alerts(
    risk_level: Optional[str] = None,
    acknowledged: Optional[bool] = None,
    limit: int = Query(50, le=500),
    offset: int = 0,
    db: Session = Depends(get_db),
):
    q = db.query(Alert)
    if risk_level:
        q = q.filter(Alert.risk_level == risk_level)
    if acknowledged is not None:
        q = q.filter(Alert.acknowledged == acknowledged)
    return q.order_by(Alert.created_at.desc()).offset(offset).limit(limit).all()


@app.patch("/alerts/{alert_id}/ack", response_model=schemas.AlertOut)
def acknowledge_alert(alert_id: int, db: Session = Depends(get_db)):
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if not alert:
        raise HTTPException(404, "Alert not found")
    alert.acknowledged = True
    db.commit()
    db.refresh(alert)
    return alert
