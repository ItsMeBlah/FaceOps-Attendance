from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any, Generic, TypeVar

from confluent_kafka import Consumer, KafkaException
from pydantic import BaseModel

from app.config import settings
from app.kafka_messaging.schemas import (
    FaceStorageRequestEvent,
    InferenceResultEvent,
)
from app.kafka_messaging.topics import (
    ALERT_WORKER_GROUP,
    DATABASE_WRITER_GROUP,
    FACE_STORAGE_REQUESTS_TOPIC,
    FACE_STORAGE_WORKER_GROUP,
    INFERENCE_RESULTS_TOPIC,
)

logger = logging.getLogger(__name__)
EventT = TypeVar("EventT", bound=BaseModel)


def _model_validate(model_class: type[EventT], payload: dict[str, Any]) -> EventT:
    if hasattr(model_class, "model_validate"):
        return model_class.model_validate(payload)
    return model_class.parse_obj(payload)


class KafkaEventConsumer(Generic[EventT]):
    """Reusable consumer for backend workers, not frontend/API request handlers."""

    def __init__(
        self,
        topics: list[str],
        group_id: str,
        event_model: type[EventT],
        handler: Callable[[EventT], None],
        bootstrap_servers: str | None = None,
        auto_offset_reset: str = "earliest",
    ) -> None:
        self.topics = topics
        self.group_id = group_id
        self.event_model = event_model
        self.handler = handler
        self.bootstrap_servers = bootstrap_servers or settings.kafka_bootstrap_servers
        self.auto_offset_reset = auto_offset_reset
        self._consumer: Consumer | None = None

    @property
    def consumer(self) -> Consumer:
        if self._consumer is None:
            logger.info(
                "Initializing Kafka consumer topics=%s group_id=%s bootstrap_servers=%s",
                self.topics,
                self.group_id,
                self.bootstrap_servers,
            )
            self._consumer = Consumer(
                {
                    "bootstrap.servers": self.bootstrap_servers,
                    "group.id": self.group_id,
                    "auto.offset.reset": self.auto_offset_reset,
                    "enable.auto.commit": False,
                }
            )
            self._consumer.subscribe(self.topics)
        return self._consumer

    def poll_once(self, timeout: float = 1.0) -> bool:
        message = self.consumer.poll(timeout)
        if message is None:
            return False
        if message.error():
            raise KafkaException(message.error())

        payload = json.loads(message.value().decode("utf-8"))
        event = _model_validate(self.event_model, payload)
        logger.info(
            "Consumed Kafka event topic=%s partition=%s offset=%s group_id=%s event_type=%s event_id=%s correlation_id=%s",
            message.topic(),
            message.partition(),
            message.offset(),
            self.group_id,
            getattr(event, "event_type", None),
            getattr(event, "event_id", None),
            getattr(event, "correlation_id", None),
        )
        self.handler(event)
        self.consumer.commit(message=message, asynchronous=False)
        logger.info(
            "Committed Kafka event topic=%s partition=%s offset=%s group_id=%s event_id=%s",
            message.topic(),
            message.partition(),
            message.offset(),
            self.group_id,
            getattr(event, "event_id", None),
        )
        return True

    def consume_forever(self, timeout: float = 1.0) -> None:
        logger.info(
            "Starting Kafka consumer loop topics=%s group_id=%s",
            self.topics,
            self.group_id,
        )
        try:
            while True:
                self.poll_once(timeout=timeout)
        except KeyboardInterrupt:
            logger.info("Kafka consumer stopped by user.")
        finally:
            self.close()

    def close(self) -> None:
        if self._consumer is not None:
            self._consumer.close()
            self._consumer = None


def build_database_writer_consumer(
    handler: Callable[[InferenceResultEvent], None],
    group_id: str = DATABASE_WRITER_GROUP,
) -> KafkaEventConsumer[InferenceResultEvent]:
    return KafkaEventConsumer(
        topics=[INFERENCE_RESULTS_TOPIC],
        group_id=group_id,
        event_model=InferenceResultEvent,
        handler=handler,
    )


def build_alert_worker_consumer(
    handler: Callable[[InferenceResultEvent], None],
    group_id: str = ALERT_WORKER_GROUP,
) -> KafkaEventConsumer[InferenceResultEvent]:
    return KafkaEventConsumer(
        topics=[INFERENCE_RESULTS_TOPIC],
        group_id=group_id,
        event_model=InferenceResultEvent,
        handler=handler,
    )


def build_face_storage_worker_consumer(
    handler: Callable[[FaceStorageRequestEvent], None],
    group_id: str = FACE_STORAGE_WORKER_GROUP,
) -> KafkaEventConsumer[FaceStorageRequestEvent]:
    return KafkaEventConsumer(
        topics=[FACE_STORAGE_REQUESTS_TOPIC],
        group_id=group_id,
        event_model=FaceStorageRequestEvent,
        handler=handler,
    )
