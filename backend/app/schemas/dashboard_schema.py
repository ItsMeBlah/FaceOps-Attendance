from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class DashboardTotals(BaseModel):
    registered_users: int = 0
    active_today: int = 0
    active_selected_day: int = 0
    emotion_total: int = 0
    sessions: int = 0
    observations: int = 0
    average_confidence: float = 0.0


class DailyDashboardStats(BaseModel):
    date: str
    active_users: int = 0
    sessions: int = 0
    unique_users: int = 0
    observations: int = 0
    average_confidence: float = 0.0
    emotion_total: int = 0
    first_seen: datetime | None = None
    last_seen: datetime | None = None


class PersonDashboardStats(BaseModel):
    user_id: str
    user_name: str
    avatar_url: str | None = None
    session_count: int = 0
    today_sessions: int = 0
    observation_count: int = 0
    today_observations: int = 0
    average_confidence: float = 0.0
    highest_confidence: float = 0.0
    first_seen: datetime | None = None
    last_seen: datetime | None = None
    dominant_emotion: str | None = None
    emotion_total: int = 0
    emotions: dict[str, int] = Field(default_factory=dict)


class SelectedDayDashboardStats(BaseModel):
    date: str
    active_users: int = 0
    emotion_total: int = 0
    emotions: dict[str, int] = Field(default_factory=dict)


class DashboardSummaryResponse(BaseModel):
    generated_at: datetime
    timezone: str = "UTC"
    days: int
    selected_date: str
    selected_day: SelectedDayDashboardStats
    totals: DashboardTotals
    daily: list[DailyDashboardStats]
    people: list[PersonDashboardStats]


class UserSessionStats(BaseModel):
    session_id: str
    first_seen: datetime | None = None
    last_seen: datetime | None = None
    duration_seconds: float = 0.0
    confidence: float = 0.0
    highest_confidence: float = 0.0
    observation_count: int = 0


class DashboardUserDetailResponse(BaseModel):
    generated_at: datetime
    selected_date: str | None = None
    user: PersonDashboardStats
    recent_sessions: list[UserSessionStats]
