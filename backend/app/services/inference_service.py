from __future__ import annotations

import base64
from io import BytesIO
import logging
from pathlib import Path

from PIL import Image
from uuid6 import uuid7

from app.kafka_messaging.producer import KafkaEventProducer
from app.kafka_messaging.schemas import (
    EmotionPayload,
    FaceStorageRequestEvent,
    ImagePayload,
    InferenceFaceResult,
    InferenceResultEvent,
    LivenessPayload,
    NormalizedBBox,
    NormalizedPoint,
    RecognitionPayload,
)
from app.services.anti_spoofing_service import AntiSpoofingService
from app.services.emotion_service import EmotionService
from app.services.face_detection_service import FaceDetectionService
from app.services.verification_service import VerificationService

from app.schemas.pipeline_schema import FaceResult, InferenceResult

logger = logging.getLogger(__name__)


class InferenceService:
    def __init__(
        self,
        liveness_threshold: float = 0.999,
        verification_threshold: float = 0.3,
        use_triton: bool | None = None,
        triton_url: str | None = None,
        weights_dir: str | Path | None = None,
        kafka_producer: KafkaEventProducer | None = None,
    ) -> None:
        self.kafka_producer = kafka_producer or KafkaEventProducer()
        self.face_detection: FaceDetectionService = FaceDetectionService(
            use_triton=use_triton,
            triton_url=triton_url,
            weights_dir=weights_dir,
        )
        self.anti_spoofing: AntiSpoofingService = AntiSpoofingService(
            use_triton=use_triton,
            triton_url=triton_url,
            weights_dir=weights_dir,
        )
        self.emotion: EmotionService = EmotionService(
            use_triton=use_triton,
            triton_url=triton_url,
            weights_dir=weights_dir,
        )
        self.verification: VerificationService = VerificationService(
            use_triton=use_triton,
            triton_url=triton_url,
            weights_dir=weights_dir,
        )

        self.liveness_threshold = liveness_threshold
        self.verification_threshold = verification_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def inference(
        self,
        image: Image.Image,
        camera_id: str = "default-camera",
        request_id: str | None = None,
    ) -> InferenceResult:
        # save frame for debugging
        # from datetime import datetime
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        # image.save(f"/home/minhcao/Swinburne/COS30082/CustomProject/-Facial-Recognition-with-Emotion-and-Liveness/debugs/inference_{timestamp}.jpg")

        event_request_id = request_id or str(uuid7())
        result = InferenceResult()
        logger.info(
            "Starting inference request_id=%s camera_id=%s image_size=%sx%s",
            event_request_id,
            camera_id,
            image.width,
            image.height,
        )

        detections = self.detect_faces(image)
        logger.info(
            "Face detection completed request_id=%s camera_id=%s detections=%s",
            event_request_id,
            camera_id,
            len(detections),
        )
        if not detections:
            self.publish_inference_result(
                camera_id=camera_id,
                request_id=event_request_id,
                faces=[],
            )
            return result

        bboxes = [d["bbox"] for d in detections]
        scores = [d["confidence"] for d in detections]
        face_crops = [d["crop"] for d in detections]
        verification_crops = [d.get("verification_crop", d["crop"]) for d in detections]
        keypoints = [d.get("keypoints", []) for d in detections]

        # Build initial FaceResult list
        face_results = []
        for bbox, score, crop, face_keypoints in zip(bboxes, scores, face_crops, keypoints):
            crop_width, crop_height = crop.size
            face_results.append(
                FaceResult(
                    bbox=bbox,
                    detection_score=score,
                    crop_width=crop_width,
                    crop_height=crop_height,
                    keypoints=face_keypoints,
                )
            )

        # Batch emotion on all faces (for testing)
        face_results = self.run_emotion_batch(face_crops, face_results)

        # Batch anti-spoofing → filter live faces
        face_results = self.run_anti_spoofing_batch(face_crops, face_results)
        live_idx = [i for i, fr in enumerate(face_results) if fr.is_live]
        logger.info(
            "Liveness completed request_id=%s camera_id=%s live_faces=%s total_faces=%s",
            event_request_id,
            camera_id,
            len(live_idx),
            len(face_results),
        )

        if not live_idx:
            result.faces = face_results
            self.publish_inference_result(
                camera_id=camera_id,
                request_id=event_request_id,
                faces=self.build_kafka_faces(face_results),
            )
            return result

        # live_crops = [face_crops[i] for i in live_idx]
        live_verification_crops = [verification_crops[i] for i in live_idx]

        # Batch verification on live faces only
        live_results = [face_results[i] for i in live_idx]
        # live_results = self.run_verification_batch(live_crops, live_results)
        live_results = self.run_verification_batch(live_verification_crops, live_results)
        verified_count = sum(1 for face_result in live_results if face_result.verified)
        logger.info(
            "Verification completed request_id=%s camera_id=%s verified_faces=%s live_faces=%s",
            event_request_id,
            camera_id,
            verified_count,
            len(live_results),
        )

        # Merge back and collect attendance
        kafka_faces = self.build_kafka_faces(face_results)
        for i, fr in zip(live_idx, live_results):
            face_results[i] = fr
            kafka_faces[i] = self.build_kafka_face(fr)
            if fr.verified and fr.employee_id:
                result.attendance_triggered.append(fr.employee_id)
                logger.info(
                    "Publishing face storage request request_id=%s camera_id=%s face_id=%s user_id=%s user_name=%s raw_size=%sx%s aligned_size=%sx%s",
                    event_request_id,
                    camera_id,
                    kafka_faces[i].face_id,
                    fr.employee_id,
                    fr.employee_name,
                    face_crops[i].width,
                    face_crops[i].height,
                    verification_crops[i].width,
                    verification_crops[i].height,
                )
                self.publish_face_storage_request(
                    camera_id=camera_id,
                    request_id=event_request_id,
                    face_id=kafka_faces[i].face_id,
                    face_result=fr,
                    bbox=bboxes[i],
                    keypoints=keypoints[i],
                    raw_face=face_crops[i],
                    aligned_face=verification_crops[i],
                )

        result.faces = face_results
        self.publish_inference_result(
            camera_id=camera_id,
            request_id=event_request_id,
            faces=kafka_faces,
        )
        logger.info(
            "Inference completed request_id=%s camera_id=%s faces=%s attendance_triggered=%s",
            event_request_id,
            camera_id,
            len(face_results),
            result.attendance_triggered,
        )

        return result

    def register_inference(
        self,
        images: Image.Image | list[Image.Image],
        person_id: str,
        person_name: str | None = None,
    ) -> dict:
        if isinstance(images, Image.Image):
            images = [images]
        if not images:
            raise ValueError("At least one registration image is required.")

        verification_crops: list[Image.Image] = []
        for image in images:
            detections = self.detect_faces(image)
            if not detections:
                raise ValueError("No face detected in one of the registration images.")

            best_detection = max(detections, key=lambda item: item["confidence"])
            verification_crops.append(
                best_detection.get("verification_crop", best_detection["crop"])
            )

        return self.verification.register(
            verification_crops,
            person_id,
            person_name=person_name,
        )

    # ------------------------------------------------------------------
    # Batch pipeline steps
    # ------------------------------------------------------------------

    def detect_faces(self, image: Image.Image) -> list[dict]:
        # Returns list[{"bbox": (x, y, w, h), "confidence": float, "crop": Image.Image, "verification_crop": Image.Image, "keypoints": [(x, y), ...]}]
        return self.face_detection.detect(image)

    def build_kafka_faces(self, face_results: list[FaceResult]) -> list[InferenceFaceResult]:
        return [self.build_kafka_face(face_result) for face_result in face_results]

    def build_kafka_face(self, face_result: FaceResult) -> InferenceFaceResult:
        return InferenceFaceResult(
            bbox=self.kafka_bbox(face_result.bbox),
            keypoints=self.kafka_keypoints(face_result.keypoints),
            recognition=RecognitionPayload(
                user_id=face_result.employee_id,
                user_name=face_result.employee_name,
                matched=face_result.verified,
                confidence=face_result.similarity,
            ),
            emotion=EmotionPayload(
                label=face_result.emotion,
                confidence=face_result.emotion_score,
            ),
            liveness=LivenessPayload(
                label="real" if face_result.is_live else "spoof",
                confidence=face_result.liveness_score,
            ),
        )

    def publish_inference_result(
        self,
        camera_id: str,
        request_id: str,
        faces: list[InferenceFaceResult],
    ) -> None:
        try:
            event = InferenceResultEvent(
                camera_id=camera_id,
                request_id=request_id,
                correlation_id=request_id,
                faces=faces,
            )
            logger.info(
                "Publishing inference result event request_id=%s camera_id=%s faces=%s",
                request_id,
                camera_id,
                len(faces),
            )
            self.kafka_producer.send_inference_result(event)
        except Exception:
            logger.exception("Failed to publish inference result for camera %s", camera_id)

    def publish_face_storage_request(
        self,
        camera_id: str,
        request_id: str,
        face_id: str,
        face_result: FaceResult,
        bbox: tuple[float, float, float, float],
        keypoints: list[tuple[float, float]],
        raw_face: Image.Image,
        aligned_face: Image.Image | None = None,
    ) -> None:
        try:
            event = FaceStorageRequestEvent(
                camera_id=camera_id,
                request_id=request_id,
                correlation_id=request_id,
                face_id=face_id,
                user_id=face_result.employee_id,
                user_name=face_result.employee_name,
                raw_face=self.image_payload(raw_face),
                aligned_face=self.image_payload(aligned_face) if aligned_face else None,
                bbox=self.kafka_bbox(bbox),
                keypoints=self.kafka_keypoints(keypoints),
                metadata={
                    "emotion": face_result.emotion,
                    "emotion_confidence": face_result.emotion_score,
                    "liveness_confidence": face_result.liveness_score,
                    "recognition_confidence": face_result.similarity,
                },
            )
            logger.info(
                "Publishing face storage event request_id=%s camera_id=%s face_id=%s user_id=%s raw_bytes_base64=%s aligned_bytes_base64=%s",
                request_id,
                camera_id,
                face_id,
                face_result.employee_id,
                len(event.raw_face.data_base64),
                len(event.aligned_face.data_base64) if event.aligned_face else 0,
            )
            self.kafka_producer.send_face_storage_request(event)
        except Exception:
            logger.exception("Failed to publish face storage request for camera %s", camera_id)

    @staticmethod
    def kafka_bbox(bbox: tuple[float, float, float, float]) -> NormalizedBBox:
        return NormalizedBBox(x=bbox[0], y=bbox[1], w=bbox[2], h=bbox[3])

    @staticmethod
    def kafka_keypoints(keypoints: list[tuple[float, float]]) -> list[NormalizedPoint]:
        return [NormalizedPoint(x=point[0], y=point[1]) for point in keypoints]

    @staticmethod
    def image_payload(image: Image.Image) -> ImagePayload:
        output = BytesIO()
        image.convert("RGB").save(output, format="JPEG")
        return ImagePayload(
            content_type="image/jpeg",
            data_base64=base64.b64encode(output.getvalue()).decode("ascii"),
            width=image.width,
            height=image.height,
        )

    def run_anti_spoofing_batch(
        self, face_crops: list[Image.Image], face_results: list[FaceResult]
    ) -> list[FaceResult]:
        predictions = self.anti_spoofing.predict(face_crops)
        for fr, pred in zip(face_results, predictions):
            fr.liveness_score = pred["confidence"]
            is_confident_spoof = (
                pred["label"] == "spoof"
                and pred["confidence"] >= self.liveness_threshold
            )
            fr.is_live = not is_confident_spoof
        return face_results

    def run_emotion_batch(
        self, face_crops: list[Image.Image], face_results: list[FaceResult]
    ) -> list[FaceResult]:
        predictions = self.emotion.predict(face_crops)
        for fr, pred in zip(face_results, predictions):
            fr.emotion = pred["label"]
            fr.emotion_score = pred["confidence"]
        return face_results

    def run_verification_batch(
        self, face_crops: list[Image.Image], face_results: list[FaceResult]
    ) -> list[FaceResult]:
        predictions = self.verification.verify(face_crops)
        for fr, pred in zip(face_results, predictions):
            fr.employee_id = pred["employee_id"]
            fr.employee_name = pred["employee_name"]
            fr.similarity = pred["confidence"]
            fr.verified = pred["matched"] and pred["confidence"] >= self.verification_threshold
        return face_results
