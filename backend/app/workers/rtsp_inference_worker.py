from __future__ import annotations

import argparse
from io import BytesIO
import logging
import os

from PIL import Image

from app.kafka_messaging.consumer import (
    CameraFrameMessage,
    build_rtsp_inference_worker_consumer,
)
from app.kafka_messaging.producer import kafka_producer
from app.services.inference_service import InferenceService


logger = logging.getLogger(__name__)


class RTSPInferenceWorker:
    """Consume RTSP camera frames and run the shared inference pipeline."""

    def __init__(self, inference_service: InferenceService | None = None) -> None:
        self.inference_service = inference_service or InferenceService(
            kafka_producer=kafka_producer
        )

    def handle(self, event: CameraFrameMessage) -> None:
        if not event.data:
            raise ValueError("Camera frame event has no image bytes.")

        with Image.open(BytesIO(event.data)) as image:
            frame = image.convert("RGB")

        logger.info(
            "Running RTSP inference camera_id=%s frame_id=%s frame_size=%sx%s timestamp=%s",
            event.camera_id,
            event.frame_id,
            frame.width,
            frame.height,
            event.timestamp,
        )
        self.inference_service.inference(
            frame,
            camera_id=event.camera_id,
            request_id=event.frame_id,
        )


def run_rtsp_inference_worker() -> None:
    worker = RTSPInferenceWorker()
    build_rtsp_inference_worker_consumer(worker.handle).consume_forever()


def main() -> None:
    _configure_logging()
    parser = argparse.ArgumentParser(description="Consume RTSP frame Kafka events.")
    parser.parse_args()
    logger.info("Starting RTSP inference worker")
    run_rtsp_inference_worker()


def _configure_logging() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


if __name__ == "__main__":
    main()
