from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.schemas.rtsp_schema import (
    RtspStartRequest,
    RtspStreamsResponse,
    RtspStreamStatus,
)
from app.services.rtsp_inference_service import (
    RTSPStreamSnapshot,
    rtsp_stream_manager,
)


router = APIRouter()


@router.post("/start", response_model=RtspStreamStatus)
def start_rtsp_stream(payload: RtspStartRequest) -> RtspStreamStatus:
    try:
        snapshot = rtsp_stream_manager.start_stream(
            rtsp_url=payload.rtsp_url,
            camera_id=payload.camera_id,
            sample_interval_seconds=payload.sample_interval_seconds,
            preview_fps=payload.preview_fps,
            frame_width=payload.frame_width,
            jpeg_quality=payload.jpeg_quality,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _response(snapshot)


@router.post("/{camera_id}/stop", response_model=RtspStreamStatus)
def stop_rtsp_stream(camera_id: str) -> RtspStreamStatus:
    try:
        snapshot = rtsp_stream_manager.stop_stream(camera_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="RTSP stream not found.") from exc
    return _response(snapshot)


@router.get("/{camera_id}/status", response_model=RtspStreamStatus)
def get_rtsp_stream_status(camera_id: str) -> RtspStreamStatus:
    snapshot = rtsp_stream_manager.get_status(camera_id)
    if snapshot is None:
        raise HTTPException(status_code=404, detail="RTSP stream not found.")
    return _response(snapshot)


@router.get("/status", response_model=RtspStreamsResponse)
def list_rtsp_stream_statuses() -> RtspStreamsResponse:
    return RtspStreamsResponse(
        streams=[_response(snapshot) for snapshot in rtsp_stream_manager.list_statuses()]
    )


@router.get("/{camera_id}/feed")
def get_rtsp_stream_feed(camera_id: str) -> StreamingResponse:
    session = rtsp_stream_manager.get_session(camera_id)
    if session is None:
        raise HTTPException(status_code=404, detail="RTSP stream not found.")
    return StreamingResponse(
        session.iter_mjpeg(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


def _response(snapshot: RTSPStreamSnapshot) -> RtspStreamStatus:
    return RtspStreamStatus(
        camera_id=snapshot.camera_id,
        status=snapshot.status,
        running=snapshot.running,
        error=snapshot.error,
        frame_count=snapshot.frame_count,
        published_count=snapshot.published_count,
        started_at=snapshot.started_at,
        stopped_at=snapshot.stopped_at,
        last_frame_at=snapshot.last_frame_at,
        feed_url=f"/api/rtsp/{snapshot.camera_id}/feed",
    )
