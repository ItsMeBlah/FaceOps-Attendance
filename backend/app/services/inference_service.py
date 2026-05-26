from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image
from polars import datetime

from app.services.face_detection_service import FaceDetectionService
from app.services.anti_spoofing_service import AntiSpoofingService
from app.services.emotion_service import EmotionService
from app.services.verification_service import VerificationService

from app.schemas.pipeline_schema import FaceResult, InferenceResult

logger = logging.getLogger(__name__)


class InferenceService:
    def __init__(
        self,
        liveness_threshold: float = 0.99,
        verification_threshold: float = 0.3,
        use_triton: bool | None = None,
        triton_url: str | None = None,
        weights_dir: str | Path | None = None,
    ) -> None:
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

    def inference(self, image: Image.Image) -> InferenceResult:
        # save frame for debugging
        # from datetime import datetime
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        # image.save(f"/home/minhcao/Swinburne/COS30082/CustomProject/-Facial-Recognition-with-Emotion-and-Liveness/debugs/inference_{timestamp}.jpg")

        result = InferenceResult()

        detections = self.detect_faces(image)
        if not detections:
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

        if not live_idx:
            result.faces = face_results
            return result

        live_crops = [face_crops[i] for i in live_idx]
        live_verification_crops = [verification_crops[i] for i in live_idx]

        # Batch verification on live faces only
        live_results = [face_results[i] for i in live_idx]
        # live_results = self.run_verification_batch(live_crops, live_results)
        live_results = self.run_verification_batch(live_verification_crops, live_results)

        # Merge back and collect attendance
        for i, fr in zip(live_idx, live_results):
            face_results[i] = fr
            if fr.verified and fr.employee_id:
                result.attendance_triggered.append(fr.employee_id)

        result.faces = face_results

        print(result)

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
