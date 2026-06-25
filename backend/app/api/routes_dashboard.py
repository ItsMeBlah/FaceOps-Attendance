from __future__ import annotations

from collections import defaultdict
from datetime import date as Date, datetime, time, timedelta, timezone
from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Query

from app.database import database
from app.logging.logger import ResultLogger
from app.schemas.dashboard_schema import (
    DailyDashboardStats,
    DashboardSummaryResponse,
    DashboardTotals,
    DashboardUserDetailResponse,
    PersonDashboardStats,
    SelectedDayDashboardStats,
    UserSessionStats,
)


router = APIRouter()

USERS_COLLECTION = "users"
VERIFICATION_COLLECTION = "verification_logs"
EMOTION_COLLECTION = "emotion_stats"
DAILY_EMOTION_COLLECTION = "daily_emotion_stats"


@router.get("/summary", response_model=DashboardSummaryResponse)
def dashboard_summary(
    days: Annotated[int, Query(ge=1, le=90)] = 14,
    selected_date: Annotated[str | None, Query(alias="date")] = None,
) -> DashboardSummaryResponse:
    try:
        now = _utc_now()
        today_start = _start_of_utc_day(now)
        chart_start = today_start - timedelta(days=days - 1)
        selected_day = _parse_date(selected_date) if selected_date else now.date()
        selected_key = selected_day.isoformat()

        users = _load_users()
        sessions = _load_sessions({})
        recent_sessions = [
            session
            for session in sessions
            if (_as_utc(session.get("last_seen")) or datetime.min.replace(tzinfo=timezone.utc)) >= chart_start
        ]
        selected_sessions = _filter_sessions_by_day(sessions, selected_day)
        today_sessions = _filter_sessions_by_day(sessions, now.date())
        daily_emotions_by_date = _load_daily_emotions_by_date(
            chart_start.date().isoformat(),
            (chart_start + timedelta(days=days - 1)).date().isoformat(),
        )
        selected_emotion_by_user = _load_daily_emotions_by_user(selected_key)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Dashboard data unavailable: {exc}")

    selected_emotions = _sum_emotions(selected_emotion_by_user.values())
    daily = _build_daily_stats(recent_sessions, daily_emotions_by_date, chart_start, days)
    people = _build_people_stats(
        users=users,
        sessions=selected_sessions,
        emotion_by_user=selected_emotion_by_user,
        today_start=today_start,
        include_empty=False,
    )
    totals = _build_totals(
        users=users,
        sessions=sessions,
        today_sessions=today_sessions,
        selected_sessions=selected_sessions,
        selected_emotions=selected_emotions,
    )

    return DashboardSummaryResponse(
        generated_at=now,
        days=days,
        selected_date=selected_key,
        selected_day=SelectedDayDashboardStats(
            date=selected_key,
            active_users=_unique_user_count(selected_sessions),
            emotion_total=sum(selected_emotions.values()),
            emotions=selected_emotions,
        ),
        totals=totals,
        daily=daily,
        people=people,
    )


@router.get("/users/{user_id}", response_model=DashboardUserDetailResponse)
def dashboard_user_detail(
    user_id: str,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    selected_date: Annotated[str | None, Query(alias="date")] = None,
) -> DashboardUserDetailResponse:
    try:
        now = _utc_now()
        user = database.find_one(USERS_COLLECTION, {"_id": user_id})
        if user is None:
            raise HTTPException(status_code=404, detail="User not found.")

        all_sessions = list(
            database.collection(VERIFICATION_COLLECTION)
            .find({"user_id": user_id})
            .sort("last_seen", -1)
        )
        if selected_date:
            selected_day = _parse_date(selected_date)
            sessions = _filter_sessions_by_day(all_sessions, selected_day)[:limit]
            emotions = _load_daily_emotions_by_user(selected_day.isoformat()).get(user_id, {})
            selected_key = selected_day.isoformat()
        else:
            sessions = all_sessions[:limit]
            emotions = database.find_one(EMOTION_COLLECTION, {"_id": user_id}) or {}
            selected_key = None
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Dashboard data unavailable: {exc}")

    person = _build_person_stats(
        user=user,
        sessions=sessions,
        emotions=emotions,
        today_start=_start_of_utc_day(now),
    )
    recent_sessions = [_session_stats(session) for session in sessions]

    return DashboardUserDetailResponse(
        generated_at=now,
        selected_date=selected_key,
        user=person,
        recent_sessions=recent_sessions,
    )


def _load_users() -> list[dict[str, Any]]:
    return list(database.collection(USERS_COLLECTION).find({"is_guest": {"$ne": True}}))


def _load_sessions(query: dict[str, Any]) -> list[dict[str, Any]]:
    return list(database.collection(VERIFICATION_COLLECTION).find(query))


def _load_daily_emotions_by_date(
    start_key: str,
    end_key: str,
) -> dict[str, list[dict[str, Any]]]:
    documents = database.collection(DAILY_EMOTION_COLLECTION).find(
        {"date": {"$gte": start_key, "$lte": end_key}}
    )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for document in documents:
        grouped[str(document.get("date", ""))].append(document)
    return grouped


def _load_daily_emotions_by_user(date_key: str) -> dict[str, dict[str, Any]]:
    return {
        str(document.get("user_id")): document
        for document in database.collection(DAILY_EMOTION_COLLECTION).find({"date": date_key})
        if document.get("user_id")
    }


def _build_daily_stats(
    sessions: list[dict[str, Any]],
    daily_emotions_by_date: dict[str, list[dict[str, Any]]],
    start: datetime,
    days: int,
) -> list[DailyDashboardStats]:
    daily: dict[str, dict[str, Any]] = {
        (start + timedelta(days=offset)).date().isoformat(): {
            "date": (start + timedelta(days=offset)).date().isoformat(),
            "sessions": 0,
            "users": set(),
            "observations": 0,
            "confidence_sum": 0.0,
            "emotion_total": 0,
            "first_seen": None,
            "last_seen": None,
        }
        for offset in range(days)
    }

    for session in sessions:
        last_seen = _as_utc(session.get("last_seen"))
        if last_seen is None:
            continue
        key = last_seen.date().isoformat()
        if key not in daily:
            continue

        bucket = daily[key]
        bucket["sessions"] += 1
        bucket["users"].add(str(session.get("user_id", "")))
        observations = int(session.get("observation_count") or 0)
        bucket["observations"] += observations
        bucket["confidence_sum"] += float(session.get("confidence") or 0.0)

        first_seen = _as_utc(session.get("first_seen"))
        bucket["first_seen"] = _min_datetime(bucket["first_seen"], first_seen)
        bucket["last_seen"] = _max_datetime(bucket["last_seen"], last_seen)

    for date_key, documents in daily_emotions_by_date.items():
        if date_key in daily:
            daily[date_key]["emotion_total"] = sum(_sum_emotions(documents).values())

    return [
        DailyDashboardStats(
            date=bucket["date"],
            active_users=len({user_id for user_id in bucket["users"] if user_id}),
            sessions=bucket["sessions"],
            unique_users=len({user_id for user_id in bucket["users"] if user_id}),
            observations=bucket["observations"],
            average_confidence=_safe_average(bucket["confidence_sum"], bucket["sessions"]),
            emotion_total=bucket["emotion_total"],
            first_seen=bucket["first_seen"],
            last_seen=bucket["last_seen"],
        )
        for bucket in daily.values()
    ]


def _build_people_stats(
    users: list[dict[str, Any]],
    sessions: list[dict[str, Any]],
    emotion_by_user: dict[str, dict[str, Any]],
    today_start: datetime,
    include_empty: bool,
) -> list[PersonDashboardStats]:
    sessions_by_user: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for session in sessions:
        sessions_by_user[str(session.get("user_id", ""))].append(session)

    people = []
    for user in users:
        user_id = str(user.get("_id"))
        user_sessions = sessions_by_user.get(user_id, [])
        emotions = emotion_by_user.get(user_id, {})
        if not include_empty and not user_sessions and not int(emotions.get("total") or 0):
            continue
        people.append(
            _build_person_stats(
                user=user,
                sessions=user_sessions,
                emotions=emotions,
                today_start=today_start,
            )
        )
    return sorted(people, key=lambda person: (person.last_seen is not None, person.last_seen), reverse=True)


def _build_person_stats(
    user: dict[str, Any],
    sessions: list[dict[str, Any]],
    emotions: dict[str, Any],
    today_start: datetime,
) -> PersonDashboardStats:
    user_id = str(user.get("user_id") or user.get("_id"))
    user_name = str(user.get("user_name") or user_id)
    first_seen = None
    last_seen = None
    confidence_sum = 0.0
    highest_confidence = 0.0
    observation_count = 0
    today_sessions = 0
    today_observations = 0

    for session in sessions:
        session_first_seen = _as_utc(session.get("first_seen"))
        session_last_seen = _as_utc(session.get("last_seen"))
        first_seen = _min_datetime(first_seen, session_first_seen)
        last_seen = _max_datetime(last_seen, session_last_seen)
        confidence_sum += float(session.get("confidence") or 0.0)
        highest_confidence = max(
            highest_confidence,
            float(session.get("highest_confidence") or session.get("confidence") or 0.0),
        )
        observations = int(session.get("observation_count") or 0)
        observation_count += observations
        if session_last_seen and session_last_seen >= today_start:
            today_sessions += 1
            today_observations += observations

    emotion_counts = _emotion_counts(emotions)
    emotion_total = int(emotions.get("total") or sum(emotion_counts.values()))
    dominant_emotion = _dominant_emotion(emotion_counts)

    return PersonDashboardStats(
        user_id=user_id,
        user_name=user_name,
        avatar_url=user.get("avatar_url"),
        session_count=len(sessions),
        today_sessions=today_sessions,
        observation_count=observation_count,
        today_observations=today_observations,
        average_confidence=_safe_average(confidence_sum, len(sessions)),
        highest_confidence=highest_confidence,
        first_seen=first_seen,
        last_seen=last_seen,
        dominant_emotion=dominant_emotion,
        emotion_total=emotion_total,
        emotions=emotion_counts,
    )


def _build_totals(
    users: list[dict[str, Any]],
    sessions: list[dict[str, Any]],
    today_sessions: list[dict[str, Any]],
    selected_sessions: list[dict[str, Any]],
    selected_emotions: dict[str, int],
) -> DashboardTotals:
    observation_count = sum(int(session.get("observation_count") or 0) for session in sessions)
    confidence_sum = sum(float(session.get("confidence") or 0.0) for session in sessions)
    return DashboardTotals(
        registered_users=len(users),
        active_today=_unique_user_count(today_sessions),
        active_selected_day=_unique_user_count(selected_sessions),
        emotion_total=sum(selected_emotions.values()),
        sessions=len(sessions),
        observations=observation_count,
        average_confidence=_safe_average(confidence_sum, len(sessions)),
    )


def _session_stats(session: dict[str, Any]) -> UserSessionStats:
    first_seen = _as_utc(session.get("first_seen"))
    last_seen = _as_utc(session.get("last_seen"))
    duration = 0.0
    if first_seen and last_seen:
        duration = max(0.0, (last_seen - first_seen).total_seconds())

    return UserSessionStats(
        session_id=str(session.get("_id")),
        first_seen=first_seen,
        last_seen=last_seen,
        duration_seconds=duration,
        confidence=float(session.get("confidence") or 0.0),
        highest_confidence=float(session.get("highest_confidence") or session.get("confidence") or 0.0),
        observation_count=int(session.get("observation_count") or 0),
    )


def _filter_sessions_by_day(
    sessions: list[dict[str, Any]],
    day: Date,
) -> list[dict[str, Any]]:
    return [
        session
        for session in sessions
        if (last_seen := _as_utc(session.get("last_seen"))) is not None
        and last_seen.date() == day
    ]


def _unique_user_count(sessions: list[dict[str, Any]]) -> int:
    return len({
        str(session.get("user_id", ""))
        for session in sessions
        if session.get("user_id")
    })


def _sum_emotions(documents: list[dict[str, Any]] | Any) -> dict[str, int]:
    totals = {emotion: 0 for emotion in ResultLogger.EMOTIONS}
    for document in documents:
        for emotion, count in _emotion_counts(document).items():
            totals[emotion] += count
    return totals


def _emotion_counts(document: dict[str, Any]) -> dict[str, int]:
    source = document.get("emotions") if isinstance(document, dict) else {}
    source = source if isinstance(source, dict) else {}
    return {
        emotion: int(source.get(emotion, 0) or 0)
        for emotion in ResultLogger.EMOTIONS
    }


def _dominant_emotion(emotions: dict[str, int]) -> str | None:
    if not emotions or max(emotions.values(), default=0) <= 0:
        return None
    return max(emotions, key=lambda emotion: emotions[emotion])


def _parse_date(value: str) -> Date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="date must use YYYY-MM-DD format.") from exc


def _safe_average(total: float, count: int) -> float:
    if count <= 0:
        return 0.0
    return float(total) / count


def _as_utc(value: Any) -> datetime | None:
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _start_of_utc_day(value: datetime) -> datetime:
    value = _as_utc(value) or _utc_now()
    return datetime.combine(value.date(), time.min, tzinfo=timezone.utc)


def _min_datetime(current: datetime | None, candidate: datetime | None) -> datetime | None:
    if candidate is None:
        return current
    if current is None or candidate < current:
        return candidate
    return current


def _max_datetime(current: datetime | None, candidate: datetime | None) -> datetime | None:
    if candidate is None:
        return current
    if current is None or candidate > current:
        return candidate
    return current
