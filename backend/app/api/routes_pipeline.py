from fastapi import APIRouter, File, Header, HTTPException, UploadFile
from PIL import Image
from uuid6 import uuid7

from app.kafka_messaging.producer import kafka_producer
from app.schemas.anti_spoofing_schema import AntiSpoofingResult
from app.schemas.common_schema import DetectedFace, NormalizedBox, NormalizedPoint
from app.schemas.emotion_schema import EmotionResult
from app.schemas.pipeline_schema import FaceAnalysis, FrameAnalysisResponse
from app.schemas.verification_schema import RecognitionResult
from app.services.inference_service import InferenceResult, InferenceService
from app.utils.preprocess import load_image_from_bytes

router = APIRouter()
inference_service = InferenceService(kafka_producer=kafka_producer)


def _load_valid_image(contents: bytes) -> tuple[Image.Image, int, int]:
    image = load_image_from_bytes(contents)
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError("Invalid image size")
    return image, width, height


@router.post("/frame", response_model=FrameAnalysisResponse)
async def analyze_frame(
    file: UploadFile = File(...),
    camera_id: str = Header("default-camera", alias="X-Camera-ID"),
) -> FrameAnalysisResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image uploads are supported.")

    try:
        contents = await file.read()
        image, width, height = _load_valid_image(contents)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    request_id = str(uuid7())
    result: InferenceResult = inference_service.inference(
        image,
        camera_id=camera_id,
        request_id=request_id,
    )

    face_results = [
        FaceAnalysis(
            face=DetectedFace(
                bbox=NormalizedBox(x=fr.bbox[0], y=fr.bbox[1], w=fr.bbox[2], h=fr.bbox[3]),
                detection_confidence=fr.detection_score,
                crop_width=fr.crop_width,
                crop_height=fr.crop_height,
                keypoints=[
                    NormalizedPoint(x=point[0], y=point[1])
                    for point in fr.keypoints
                ],
            ),
            emotion=EmotionResult(label=fr.emotion, confidence=fr.emotion_score),
            anti_spoofing=AntiSpoofingResult(
                label="real" if fr.is_live else "spoof",
                confidence=fr.liveness_score,
            ),
            recognition=RecognitionResult(
                label=fr.employee_name or "unknown",
                confidence=fr.similarity,
                matched=fr.verified,
            ),
        )
        for fr in result.faces
    ]

    return FrameAnalysisResponse(image_width=width, image_height=height, faces=face_results)
