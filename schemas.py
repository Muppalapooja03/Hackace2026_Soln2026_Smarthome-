"""
Pydantic schemas — request/response shapes for the API.
Kept separate from the SQLAlchemy models (db_models.py) so the API
contract can evolve independently of the storage schema.
"""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class EventIn(BaseModel):
    """What a device/sensor sends in."""
    device_id: str = Field(..., examples=["front_door"])
    event_type: str = Field(..., examples=["open"])
    timestamp: Optional[datetime] = None  # server stamps "now" if omitted


class EventOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    event_uuid: str
    timestamp: datetime
    device_id: str
    event_type: str
    hour: int
    day_of_week: int
    is_weekend: int
    analyzed: bool


class AnalyzeRequest(BaseModel):
    """Analyze specific events by id, or omit to analyze everything pending."""
    event_ids: Optional[List[int]] = None


class AnalyzeResultOut(BaseModel):
    event_id: int
    device_id: str
    event_type: str
    anomaly_score: float
    is_anomaly: bool
    risk_level: str
    risk_reasons: List[str] = []


class EventScored(EventOut):
    """An event joined with its anomaly result, if it's been analyzed."""
    anomaly_score: Optional[float] = None
    is_anomaly: Optional[bool] = None
    risk_level: Optional[str] = None


class AlertOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    event_id: int
    risk_level: str
    message: str
    acknowledged: bool
    created_at: datetime
