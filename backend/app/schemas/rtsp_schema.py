from __future__ import annotations

from pydantic import BaseModel, Field


class RtspStartRequest(BaseModel):
    rtsp_url: str = Field(..., min_length=7)
    camera_id: str | None = None
    sample_interval_seconds: float | None = Field(default=None, ge=0.1)
    preview_fps: float | None = Field(default=None, ge=0.5, le=30)
    frame_width: int | None = Field(default=None, ge=64)
    jpeg_quality: int | None = Field(default=None, ge=30, le=95)


class RtspStreamStatus(BaseModel):
    camera_id: str
    status: str
    running: bool
    error: str | None = None
    frame_count: int = 0
    published_count: int = 0
    started_at: str | None = None
    stopped_at: str | None = None
    last_frame_at: str | None = None
    feed_url: str | None = None


class RtspStreamsResponse(BaseModel):
    streams: list[RtspStreamStatus]
