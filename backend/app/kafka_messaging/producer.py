from __future__ import annotations

import json
import logging
from typing import Any

from confluent_kafka import Producer
from pydantic import BaseModel

from app.config import settings
from app.kafka_messaging.schemas import (
    FaceStorageRequestEvent,
    InferenceResultEvent,
)
from app.kafka_messaging.topics import (
    CAMERA_FRAME_EVENTS_TOPIC,
    FACE_STORAGE_REQUESTS_TOPIC,
    INFERENCE_RESULTS_TOPIC,
)

logger = logging.getLogger(__name__)


def _event_field(event: BaseModel | dict[str, Any], field_name: str) -> Any:
    if isinstance(event, BaseModel):
        return getattr(event, field_name, None)
    return event.get(field_name)


def _to_json_bytes(event: BaseModel | dict[str, Any]) -> bytes:
    if isinstance(event, BaseModel):
        if hasattr(event, "model_dump_json"):
            return event.model_dump_json().encode("utf-8")
        return event.json().encode("utf-8")
    return json.dumps(event, default=str, separators=(",", ":")).encode("utf-8")


class KafkaEventProducer:
    """Producer for backend background write events only."""

    def __init__(
        self,
        bootstrap_servers: str | None = None,
        client_id: str | None = None,
    ) -> None:
        self.bootstrap_servers = bootstrap_servers or settings.kafka_bootstrap_servers
        self.client_id = client_id or settings.kafka_client_id
        self._producer: Producer | None = None

    @property
    def producer(self) -> Producer:
        if self._producer is None:
            logger.info(
                "Initializing Kafka producer bootstrap_servers=%s client_id=%s",
                self.bootstrap_servers,
                self.client_id,
            )
            self._producer = Producer(
                {
                    "bootstrap.servers": self.bootstrap_servers,
                    "client.id": self.client_id,
                    "enable.idempotence": True,
                    "acks": "all",
                }
            )
        return self._producer

    def send(
        self,
        topic: str,
        event: BaseModel | dict[str, Any],
        key: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        value = _to_json_bytes(event)
        event_id = _event_field(event, "event_id")
        event_type = _event_field(event, "event_type")
        correlation_id = _event_field(event, "correlation_id")
        logger.info(
            "Queueing Kafka event topic=%s key=%s event_type=%s event_id=%s correlation_id=%s bytes=%s",
            topic,
            key,
            event_type,
            event_id,
            correlation_id,
            len(value),
        )
        self.producer.produce(
            topic=topic,
            key=key.encode("utf-8") if key else None,
            value=value,
            headers=list((headers or {}).items()),
            callback=self._delivery_report,
        )
        self.producer.poll(0)

    def send_inference_result(self, event: InferenceResultEvent) -> None:
        """Model path -> DB writer/dashboard realtime/alert workers."""
        self.send(INFERENCE_RESULTS_TOPIC, event, key=event.camera_id)

    def send_face_storage_request(self, event: FaceStorageRequestEvent) -> None:
        """Model path -> face storage worker."""
        self.send(FACE_STORAGE_REQUESTS_TOPIC, event, key=event.camera_id)

    def send_camera_frame(
        self,
        camera_id: str,
        frame_id: str,
        frame_bytes: bytes,
        timestamp: str,
        content_type: str = "image/jpeg",
    ) -> None:
        """RTSP ingestion path -> inference workers.

        Camera frames are sent as raw JPEG bytes to avoid bloating Kafka
        messages with base64. Metadata stays in headers so consumers can route
        by camera without parsing a JSON envelope.
        """
        headers = {
            "camera_id": camera_id,
            "frame_id": frame_id,
            "timestamp": timestamp,
            "content_type": content_type,
        }
        logger.info(
            "Queueing Kafka frame topic=%s camera_id=%s frame_id=%s bytes=%s",
            CAMERA_FRAME_EVENTS_TOPIC,
            camera_id,
            frame_id,
            len(frame_bytes),
        )
        self.producer.produce(
            topic=CAMERA_FRAME_EVENTS_TOPIC,
            key=camera_id.encode("utf-8"),
            value=frame_bytes,
            headers=[(key, value.encode("utf-8")) for key, value in headers.items()],
            callback=self._delivery_report,
        )
        self.producer.poll(0)

    def flush(self, timeout: float = 5.0) -> None:
        self.producer.flush(timeout)

    def close(self) -> None:
        if self._producer is not None:
            self._producer.flush(5.0)
            self._producer = None

    @staticmethod
    def _delivery_report(error, message) -> None:
        if error is not None:
            logger.error("Kafka delivery failed: %s", error)
            return
        logger.info(
            "Kafka event delivered topic=%s partition=%s offset=%s key=%s",
            message.topic(),
            message.partition(),
            message.offset(),
            message.key().decode("utf-8") if message.key() else None,
        )


kafka_producer = KafkaEventProducer()
