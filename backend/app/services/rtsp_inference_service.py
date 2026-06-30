from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterator
from urllib.parse import urlsplit, urlunsplit

import cv2
from uuid6 import uuid7

from app.config import settings
from app.kafka_messaging.producer import KafkaEventProducer, kafka_producer


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RTSPStreamConfig:
    camera_id: str
    rtsp_url: str
    sample_interval_seconds: float = settings.rtsp_sample_interval_seconds
    preview_fps: float = settings.rtsp_preview_fps
    frame_width: int = settings.rtsp_frame_width
    jpeg_quality: int = settings.rtsp_jpeg_quality


@dataclass(frozen=True)
class RTSPStreamSnapshot:
    camera_id: str
    status: str
    running: bool
    error: str | None
    frame_count: int
    published_count: int
    started_at: str | None
    stopped_at: str | None
    last_frame_at: str | None


class RTSPStreamSession:
    """Read one RTSP stream, publish sampled JPEG frames, and expose MJPEG preview."""

    def __init__(
        self,
        config: RTSPStreamConfig,
        producer: KafkaEventProducer | None = None,
    ) -> None:
        self.config = config
        self.producer = producer or kafka_producer
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._capture: cv2.VideoCapture | None = None
        self._lock = threading.Lock()
        self._new_frame = threading.Condition(self._lock)
        self._latest_jpeg: bytes | None = None
        self._latest_frame_number = 0
        self._status = "stopped"
        self._error: str | None = None
        self._frame_count = 0
        self._published_count = 0
        self._started_at: str | None = None
        self._stopped_at: str | None = None
        self._last_frame_at: str | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return

        self._stop_event.clear()
        with self._lock:
            self._status = "starting"
            self._error = None
            self._started_at = utc_timestamp()
            self._stopped_at = None

        self._thread = threading.Thread(
            target=self._run,
            name=f"rtsp-stream-{self.config.camera_id}",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "RTSP stream thread started camera_id=%s url=%s",
            self.config.camera_id,
            redact_rtsp_url(self.config.rtsp_url),
        )

    def stop(self, timeout: float = 5.0) -> None:
        self._stop_event.set()
        capture = self._capture
        if capture is not None:
            capture.release()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        with self._lock:
            if self._status != "error":
                self._status = "stopped"
            self._stopped_at = self._stopped_at or utc_timestamp()
            self._new_frame.notify_all()
        logger.info("RTSP stream stopped camera_id=%s", self.config.camera_id)

    def snapshot(self) -> RTSPStreamSnapshot:
        with self._lock:
            return RTSPStreamSnapshot(
                camera_id=self.config.camera_id,
                status=self._status,
                running=self._status in {"starting", "running"},
                error=self._error,
                frame_count=self._frame_count,
                published_count=self._published_count,
                started_at=self._started_at,
                stopped_at=self._stopped_at,
                last_frame_at=self._last_frame_at,
            )

    def iter_mjpeg(self) -> Iterator[bytes]:
        last_frame_number = 0
        while True:
            with self._new_frame:
                self._new_frame.wait_for(
                    lambda: (
                        self._latest_frame_number != last_frame_number
                        or self._status in {"stopped", "error"}
                    ),
                    timeout=2.0,
                )
                jpeg = self._latest_jpeg
                frame_number = self._latest_frame_number
                status = self._status

            if jpeg and frame_number != last_frame_number:
                last_frame_number = frame_number
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Cache-Control: no-store\r\n\r\n"
                    + jpeg
                    + b"\r\n"
                )
                continue

            if status in {"stopped", "error"}:
                break

    def _run(self) -> None:
        capture: cv2.VideoCapture | None = None
        try:
            capture = cv2.VideoCapture(self.config.rtsp_url)
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self._capture = capture

            if not capture.isOpened():
                raise RuntimeError("Unable to open RTSP stream.")
            if self._stop_event.is_set():
                return

            with self._lock:
                self._status = "running"
                self._new_frame.notify_all()

            next_preview_at = 0.0
            next_publish_at = 0.0
            failed_reads = 0

            while not self._stop_event.is_set():
                ok, frame = capture.read()
                if not ok or frame is None:
                    failed_reads += 1
                    if failed_reads >= 30:
                        raise RuntimeError("RTSP stream stopped returning frames.")
                    time.sleep(0.1)
                    continue

                failed_reads = 0
                now = time.monotonic()
                frame = resize_frame(frame, self.config.frame_width)

                jpeg: bytes | None = None
                if now >= next_preview_at:
                    jpeg = encode_jpeg(frame, self.config.jpeg_quality)
                    self._set_latest_frame(jpeg)
                    next_preview_at = now + 1.0 / max(self.config.preview_fps, 0.5)

                if now >= next_publish_at:
                    if jpeg is None:
                        jpeg = encode_jpeg(frame, self.config.jpeg_quality)
                    self._publish_frame(jpeg)
                    next_publish_at = now + max(self.config.sample_interval_seconds, 0.1)

        except Exception as exc:
            logger.exception(
                "RTSP stream failed camera_id=%s url=%s",
                self.config.camera_id,
                redact_rtsp_url(self.config.rtsp_url),
            )
            with self._lock:
                self._status = "error"
                self._error = str(exc)
                self._stopped_at = utc_timestamp()
                self._new_frame.notify_all()
        finally:
            if capture is not None:
                capture.release()
            self._capture = None
            if self._stop_event.is_set():
                with self._lock:
                    self._status = "stopped"
                    self._stopped_at = self._stopped_at or utc_timestamp()
                    self._new_frame.notify_all()

    def _set_latest_frame(self, jpeg: bytes) -> None:
        with self._lock:
            self._latest_jpeg = jpeg
            self._latest_frame_number += 1
            self._frame_count += 1
            self._last_frame_at = utc_timestamp()
            self._new_frame.notify_all()

    def _publish_frame(self, jpeg: bytes) -> None:
        frame_id = str(uuid7())
        timestamp = utc_timestamp()
        self.producer.send_camera_frame(
            camera_id=self.config.camera_id,
            frame_id=frame_id,
            frame_bytes=jpeg,
            timestamp=timestamp,
            content_type="image/jpeg",
        )
        with self._lock:
            self._published_count += 1


class RTSPStreamManager:
    def __init__(self, producer: KafkaEventProducer | None = None) -> None:
        self.producer = producer or kafka_producer
        self._lock = threading.Lock()
        self._sessions: dict[str, RTSPStreamSession] = {}

    def start_stream(
        self,
        rtsp_url: str,
        camera_id: str | None = None,
        sample_interval_seconds: float | None = None,
        preview_fps: float | None = None,
        frame_width: int | None = None,
        jpeg_quality: int | None = None,
    ) -> RTSPStreamSnapshot:
        validate_rtsp_url(rtsp_url)
        clean_camera_id = normalize_camera_id(camera_id)
        config = RTSPStreamConfig(
            camera_id=clean_camera_id,
            rtsp_url=rtsp_url.strip(),
            sample_interval_seconds=sample_interval_seconds or settings.rtsp_sample_interval_seconds,
            preview_fps=preview_fps or settings.rtsp_preview_fps,
            frame_width=frame_width or settings.rtsp_frame_width,
            jpeg_quality=jpeg_quality or settings.rtsp_jpeg_quality,
        )

        with self._lock:
            existing = self._sessions.get(clean_camera_id)
            if existing is not None:
                existing.stop()
            session = RTSPStreamSession(config=config, producer=self.producer)
            self._sessions[clean_camera_id] = session

        session.start()
        return session.snapshot()

    def stop_stream(self, camera_id: str) -> RTSPStreamSnapshot:
        session = self.get_session(camera_id)
        if session is None:
            raise KeyError(camera_id)
        session.stop()
        return session.snapshot()

    def get_status(self, camera_id: str) -> RTSPStreamSnapshot | None:
        session = self.get_session(camera_id)
        return session.snapshot() if session else None

    def list_statuses(self) -> list[RTSPStreamSnapshot]:
        with self._lock:
            sessions = list(self._sessions.values())
        return [session.snapshot() for session in sessions]

    def get_session(self, camera_id: str) -> RTSPStreamSession | None:
        clean_camera_id = normalize_camera_id(camera_id)
        with self._lock:
            return self._sessions.get(clean_camera_id)

    def stop_all(self) -> None:
        with self._lock:
            sessions = list(self._sessions.values())
        for session in sessions:
            session.stop()


def validate_rtsp_url(rtsp_url: str) -> None:
    parsed = urlsplit(rtsp_url.strip())
    if parsed.scheme.lower() != "rtsp" or not parsed.netloc:
        raise ValueError("rtsp_url must be a valid rtsp:// URL.")


def normalize_camera_id(camera_id: str | None = None) -> str:
    value = (camera_id or "").strip()
    if not value:
        return f"camera-{uuid7()}"
    for char in ("/", "\\", "?", "#", "&"):
        value = value.replace(char, "-")
    return value.replace(" ", "-")


def resize_frame(frame, target_width: int):
    if target_width <= 0:
        return frame
    height, width = frame.shape[:2]
    if width <= target_width:
        return frame
    ratio = target_width / float(width)
    target_height = max(1, int(height * ratio))
    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)


def encode_jpeg(frame, quality: int) -> bytes:
    ok, encoded = cv2.imencode(
        ".jpg",
        frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)],
    )
    if not ok:
        raise RuntimeError("Failed to JPEG encode RTSP frame.")
    return encoded.tobytes()


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def redact_rtsp_url(rtsp_url: str) -> str:
    parsed = urlsplit(rtsp_url)
    if not parsed.username and not parsed.password:
        return rtsp_url
    host = parsed.hostname or ""
    if parsed.port:
        host = f"{host}:{parsed.port}"
    return urlunsplit((parsed.scheme, host, parsed.path, parsed.query, parsed.fragment))


rtsp_stream_manager = RTSPStreamManager()
