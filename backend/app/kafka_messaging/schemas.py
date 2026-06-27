from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field
from uuid6 import uuid7


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class KafkaEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid7()))
    event_type: str
    schema_version: str = "1.0"
    source: str = "faceguard"
    created_at: datetime = Field(default_factory=utc_now)
    correlation_id: str | None = None


class ImagePayload(BaseModel):
    content_type: str = "image/jpeg"
    data_base64: str
    width: int
    height: int


class NormalizedBBox(BaseModel):
    x: float
    y: float
    w: float
    h: float


class NormalizedPoint(BaseModel):
    x: float
    y: float


class RecognitionPayload(BaseModel):
    user_id: str | None = None
    user_name: str | None = None
    matched: bool = False
    confidence: float = 0.0


class EmotionPayload(BaseModel):
    label: str = ""
    confidence: float = 0.0


class LivenessPayload(BaseModel):
    label: Literal["real", "spoof"] = "spoof"
    confidence: float = 0.0


class InferenceFaceResult(BaseModel):
    face_id: str = Field(default_factory=lambda: str(uuid7()))
    bbox: NormalizedBBox
    keypoints: list[NormalizedPoint] = Field(default_factory=list)
    recognition: RecognitionPayload = Field(default_factory=RecognitionPayload)
    emotion: EmotionPayload = Field(default_factory=EmotionPayload)
    liveness: LivenessPayload = Field(default_factory=LivenessPayload)


class InferenceResultEvent(KafkaEvent):
    event_type: Literal["inference.result"] = "inference.result"
    request_id: str
    camera_id: str
    processed_at: datetime = Field(default_factory=utc_now)
    faces: list[InferenceFaceResult] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class FaceStorageRequestEvent(KafkaEvent):
    event_type: Literal["face.storage.request"] = "face.storage.request"
    request_id: str
    camera_id: str
    face_id: str = Field(default_factory=lambda: str(uuid7()))
    user_id: str | None = None
    user_name: str | None = None
    raw_face: ImagePayload
    aligned_face: ImagePayload | None = None
    bbox: NormalizedBBox
    keypoints: list[NormalizedPoint] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
