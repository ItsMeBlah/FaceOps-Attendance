from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from app.config import settings
from app.logging.logger import ResultLogger
from app.services.base_service import BaseService
from app.services.vector_store_service import VectorStoreService


class VerificationService(BaseService):
	def __init__(
		self,
		use_triton: bool | None = None,
		triton_url: str | None = None,
		weights_dir: str | Path | None = None,
		model_path: str | Path | None = None,
		match_threshold: float = 0.3,
		vector_store: VectorStoreService | None = None,
		result_logger: ResultLogger | None = None,
	) -> None:
		super().__init__(
			"resnet18_face",
			use_triton=use_triton,
			triton_url=triton_url,
			weights_dir=weights_dir,
			model_path=model_path,
		)
		self.match_threshold = match_threshold
		self.vector_store = vector_store or VectorStoreService()
		self.result_logger = result_logger or ResultLogger()
		self.max_registration_images = settings.qdrant_max_registration_images
		self.INPUT_SIZE = (128, 128)
		self.CHUNK_SIZE = 32

	def preprocess(self, image: np.ndarray | list[np.ndarray]) -> dict[str, np.ndarray]:
		images = image if isinstance(image, list) else [image]
		input_tensors = []
		for image_item in images:
			gray_image = Image.fromarray(image_item).convert("L")
			if gray_image.size != self.INPUT_SIZE:
				raise ValueError(
					f"Verification crop must be {self.INPUT_SIZE}, got {gray_image.size}. "
					"Resize/alignment should happen before VerificationService.preprocess()."
				)
			input_tensor = np.asarray(gray_image, dtype=np.float32) / 255.0
			input_tensor = (input_tensor - 0.5) / 0.5
			input_tensor = input_tensor[np.newaxis, ...]
			input_tensors.append(input_tensor)

		input_name = self._input_metadata[0]["name"] if self._input_metadata else "input_1"
		return {input_name: np.stack(input_tensors, axis=0).astype(np.float32)}

	def postprocess(self, outputs: dict[str, np.ndarray]) -> np.ndarray:
		return self._read_outputs(outputs)[0]

	def verify(self, face_images: Image.Image | list[Image.Image]) -> list[dict]:
		"""
		Output per face:
		  {
		    "employee_id":   str | None,  # DB employee ID; None if unrecognised
		    "employee_name": str,         # display name; "unknown" if unrecognised
		    "confidence":    float,       # cosine similarity score 0.0 - 1.0
		    "matched":       bool         # True if confidence >= verification threshold
		  }
		Returns a list even for a single image input.
		"""
		if isinstance(face_images, Image.Image):
			face_images = [face_images]

		results: list[dict] = []
		for embedding in self._extract_embeddings(face_images):
			match = self.vector_store.pick_majority_match(embedding)
			matched = match is not None and match.score >= self.match_threshold
			employee_id = match.employee_id if matched and match is not None else None
			employee_name = match.employee_name if matched and match is not None else "unknown"
			score = match.score if match is not None else 0.0

			results.append(
				{
					"employee_id": employee_id,
					"employee_name": employee_name,
					"label": employee_name,
					"confidence": float(score),
					"matched": matched,
				}
			)
		return results

	def register(
		self,
		face_images: Image.Image | list[Image.Image],
		person_id: str,
		person_name: str | None = None,
	) -> dict:
		"""
		Output:
		  {
		    "person_id": str,
		    "status":    "registered" | "updated"
		  }
		"""
		if isinstance(face_images, Image.Image):
			face_images = [face_images]
		if not face_images:
			raise ValueError("At least one registration image is required.")
		if len(face_images) > self.max_registration_images:
			raise ValueError(
				f"At most {self.max_registration_images} registration images are allowed."
			)

		person_id = person_id.strip()
		if not person_id:
			raise ValueError("Person ID is required.")
		person_name = (person_name or "").strip() or person_id

		status = "updated" if self.vector_store.employee_exists(person_id) else "registered"
		embeddings = self._extract_embeddings(face_images)
		self.vector_store.register_embeddings(
			employee_id=person_id,
			employee_name=person_name,
			embeddings=embeddings,
		)
		self.result_logger.upsert_registered_user(person_id, person_name)
		return {
			"person_id": person_id,
			"status": status,
			"image_count": len(embeddings),
		}

	def _extract_embeddings(self, face_images: list[Image.Image]) -> list[np.ndarray]:
		if not face_images:
			return []

		self._ensure_loaded()
		embeddings: list[np.ndarray] = []

		for start in range(0, len(face_images), self.CHUNK_SIZE):
			chunk = face_images[start:start + self.CHUNK_SIZE]
			images = [
				np.asarray(face_image.convert("RGB"), dtype=np.uint8)
				for face_image in chunk
			]
			raw_outputs = self._infer(self.preprocess(images))
			embeddings.extend(self._read_outputs(raw_outputs))

		return embeddings

	def _read_outputs(self, outputs: dict[str, np.ndarray]) -> list[np.ndarray]:
		embeddings = next(iter(outputs.values())).astype(np.float32)
		if embeddings.ndim == 1:
			embeddings = embeddings[np.newaxis, :]

		norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
		if np.any(norms == 0):
			raise ValueError("Verification model returned a zero-norm embedding.")

		normalized = embeddings / norms
		return [embedding for embedding in normalized]
