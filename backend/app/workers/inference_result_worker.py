from __future__ import annotations

import argparse
import logging
import os
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from app.database import database
from app.kafka_messaging.consumer import (
    build_alert_worker_consumer,
    build_database_writer_consumer,
)
from app.kafka_messaging.schemas import InferenceFaceResult, InferenceResultEvent
from app.logging.logger import ResultLogger


logger = logging.getLogger(__name__)

INFERENCE_EVENTS_COLLECTION = "inference_events"
ALERTS_COLLECTION = "alerts"


def handle_database_write(event: InferenceResultEvent) -> None:
    """Persist inference results into collections already used by the dashboard."""
    result_logger = ResultLogger()
    matched_faces = 0
    emotion_updates = 0

    for face in event.faces:
        recognition = face.recognition
        if not recognition.matched or not recognition.user_id:
            continue

        result_logger.log_verification(
            user_id=recognition.user_id,
            user_name=recognition.user_name,
            matched=recognition.matched,
            confidence=recognition.confidence,
        )
        matched_faces += 1

        if face.emotion.label:
            try:
                result_logger.log_emotion(
                    user_id=recognition.user_id,
                    emotion=face.emotion.label,
                    confidence=face.emotion.confidence,
                    user_name=recognition.user_name,
                )
                emotion_updates += 1
            except ValueError:
                logger.warning(
                    "Skipping unsupported emotion label event_id=%s face_id=%s label=%s",
                    event.event_id,
                    face.face_id,
                    face.emotion.label,
                )

    database.update_one(
        INFERENCE_EVENTS_COLLECTION,
        {"_id": event.event_id},
        {
            "$set": {
                "request_id": event.request_id,
                "camera_id": event.camera_id,
                "processed_at": event.processed_at,
                "created_at": event.created_at,
                "faces": [_model_to_dict(face) for face in event.faces],
                "face_count": len(event.faces),
                "matched_faces": matched_faces,
                "emotion_updates": emotion_updates,
                "metadata": event.metadata,
                "updated_at": _utc_now(),
            },
        },
        upsert=True,
    )
    logger.info(
        "Database writer stored inference result event_id=%s request_id=%s camera_id=%s faces=%s matched_faces=%s emotion_updates=%s",
        event.event_id,
        event.request_id,
        event.camera_id,
        len(event.faces),
        matched_faces,
        emotion_updates,
    )


def handle_alert(event: InferenceResultEvent) -> None:
    """Create alert records for spoof or unknown faces."""
    alerts = [_build_alert(event, face) for face in event.faces if _needs_alert(face)]
    for alert in alerts:
        alert_id = alert.pop("_id")
        database.update_one(
            ALERTS_COLLECTION,
            {"_id": alert_id},
            {"$set": alert},
            upsert=True,
        )

    logger.info(
        "Alert worker processed inference result event_id=%s camera_id=%s alerts=%s",
        event.event_id,
        event.camera_id,
        len(alerts),
    )


def run_database_writer() -> None:
    build_database_writer_consumer(handle_database_write).consume_forever()


def run_alert_worker() -> None:
    build_alert_worker_consumer(handle_alert).consume_forever()


def main() -> None:
    _configure_logging()
    parser = argparse.ArgumentParser(description="Consume inference result Kafka events.")
    parser.add_argument(
        "worker",
        choices=("database", "alert"),
        default="database",
        nargs="?",
        help="Consumer role to run.",
    )
    args = parser.parse_args()

    runners: dict[str, Callable[[], None]] = {
        "database": run_database_writer,
        "alert": run_alert_worker,
    }
    logger.info("Starting inference result worker role=%s", args.worker)
    runners[args.worker]()


def _build_alert(event: InferenceResultEvent, face: InferenceFaceResult) -> dict[str, Any]:
    alert_type = "spoof_face" if face.liveness.label != "real" else "unknown_face"
    recognition = face.recognition
    return {
        "_id": f"{event.event_id}:{face.face_id}:{alert_type}",
        "alert_type": alert_type,
        "event_id": event.event_id,
        "request_id": event.request_id,
        "camera_id": event.camera_id,
        "face_id": face.face_id,
        "user_id": recognition.user_id,
        "user_name": recognition.user_name,
        "recognition": _model_to_dict(recognition),
        "emotion": _model_to_dict(face.emotion),
        "liveness": _model_to_dict(face.liveness),
        "bbox": _model_to_dict(face.bbox),
        "keypoints": [_model_to_dict(point) for point in face.keypoints],
        "processed_at": event.processed_at,
        "created_at": event.created_at,
        "updated_at": _utc_now(),
    }


def _needs_alert(face: InferenceFaceResult) -> bool:
    return face.liveness.label != "real" or not face.recognition.matched


def _model_to_dict(model: Any) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _configure_logging() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


if __name__ == "__main__":
    main()
