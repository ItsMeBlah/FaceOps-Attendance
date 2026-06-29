from __future__ import annotations

import argparse
import base64
from io import BytesIO
import logging
import os
from datetime import datetime, timezone
from typing import Any

from PIL import Image

from app.database import database
from app.kafka_messaging.consumer import build_face_storage_worker_consumer
from app.kafka_messaging.schemas import FaceStorageRequestEvent, ImagePayload
from app.logging.image_logger import ImageLogger


logger = logging.getLogger(__name__)

FACE_STORAGE_LOGS_COLLECTION = "face_storage_logs"


class FaceStorageWorker:
    """Consume face-storage requests and persist face assets plus metadata."""

    def __init__(
        self,
        image_logger: ImageLogger | None = None,
    ) -> None:
        self.image_logger = image_logger or ImageLogger()

    def handle(self, event: FaceStorageRequestEvent) -> None:
        user_name = _storage_user_name(event)
        raw_image = _image_from_payload(event.raw_face)
        aligned_image = _image_from_payload(event.aligned_face) if event.aligned_face else None
        log_date = self.image_logger.current_log_date()
        image_index = self.image_logger.next_image_index("raw", user_name, log_date=log_date)

        logger.info(
            "Storing face images event_id=%s request_id=%s camera_id=%s face_id=%s user_id=%s user_name=%s image_index=%s",
            event.event_id,
            event.request_id,
            event.camera_id,
            event.face_id,
            event.user_id,
            user_name,
            image_index,
        )
        raw_object_name = self.image_logger.insert_raw_image(
            user_name=user_name,
            image=raw_image,
            index=image_index,
            log_date=log_date,
        )
        aligned_object_name = None
        if aligned_image is not None:
            aligned_object_name = self.image_logger.insert_aligned_image(
                user_name=user_name,
                image=aligned_image,
                index=image_index,
                log_date=log_date,
            )

        self._store_metadata(
            event=event,
            user_name=user_name,
            image_index=image_index,
            log_date=log_date,
            raw_object_name=raw_object_name,
            aligned_object_name=aligned_object_name,
        )
        logger.info(
            "Face storage request completed event_id=%s request_id=%s face_id=%s raw_object=%s aligned_object=%s",
            event.event_id,
            event.request_id,
            event.face_id,
            raw_object_name,
            aligned_object_name,
        )

    def _store_metadata(
        self,
        event: FaceStorageRequestEvent,
        user_name: str,
        image_index: int,
        log_date: str,
        raw_object_name: str,
        aligned_object_name: str | None,
    ) -> None:
        database.update_one(
            FACE_STORAGE_LOGS_COLLECTION,
            {"_id": event.event_id},
            {
                "$set": {
                    "request_id": event.request_id,
                    "camera_id": event.camera_id,
                    "face_id": event.face_id,
                    "user_id": event.user_id,
                    "user_name": user_name,
                    "log_date": log_date,
                    "image_index": image_index,
                    "raw_bucket": self.image_logger.logs_bucket,
                    "raw_object_name": raw_object_name,
                    "aligned_bucket": self.image_logger.aligned_images_bucket,
                    "aligned_object_name": aligned_object_name,
                    "bbox": _model_to_dict(event.bbox),
                    "keypoints": [_model_to_dict(point) for point in event.keypoints],
                    "metadata": event.metadata,
                    "created_at": event.created_at,
                    "updated_at": _utc_now(),
                },
            },
            upsert=True,
        )


def run_face_storage_worker() -> None:
    worker = FaceStorageWorker()
    build_face_storage_worker_consumer(worker.handle).consume_forever()


def main() -> None:
    _configure_logging()
    parser = argparse.ArgumentParser(description="Consume face storage Kafka events.")
    parser.parse_args()
    logger.info("Starting face storage worker")
    run_face_storage_worker()


def _image_from_payload(payload: ImagePayload) -> Image.Image:
    image_bytes = base64.b64decode(payload.data_base64)
    with Image.open(BytesIO(image_bytes)) as image:
        return image.convert("RGB")


def _storage_user_name(event: FaceStorageRequestEvent) -> str:
    return (event.user_name or event.user_id or "unknown").strip() or "unknown"


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
