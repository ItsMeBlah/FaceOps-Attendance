from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from app.services.base_service import BaseService


class EmotionService(BaseService):
	def __init__(
		self,
		use_triton: bool | None = None,
		triton_url: str | None = None,
		weights_dir: str | Path | None = None,
		model_path: str | Path | None = None,
	) -> None:
		super().__init__(
			"emotion",
			use_triton=use_triton,
			triton_url=triton_url,
			weights_dir=weights_dir,
			model_path=model_path,
		)
		self.INPUT_SIZE = (224, 224)
		self.NORM_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
		self.NORM_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
		self.LABELS = [
			"anger",
			"contempt",
			"disgust",
			"fear",
			"happy",
			"neutral",
			"sad",
			"surprise",
		]
		self.CHUNK_SIZE = 32

	def preprocess(self, image: np.ndarray | list[np.ndarray]) -> dict[str, np.ndarray]:
		images = image if isinstance(image, list) else [image]
		input_tensors = []
		for image_item in images:
			resized = Image.fromarray(image_item).resize(self.INPUT_SIZE, Image.BILINEAR)
			arr = np.asarray(resized, dtype=np.float32) / 255.0
			arr = (arr - self.NORM_MEAN) / self.NORM_STD
			arr = np.transpose(arr, (2, 0, 1))
			input_tensors.append(arr)

		input_name = self._input_metadata[0]["name"] if self._input_metadata else "input"
		return {input_name: np.stack(input_tensors, axis=0).astype(np.float32)}

	def postprocess(self, outputs: dict[str, np.ndarray]) -> list[dict]:
		logits = next(iter(outputs.values())).astype(np.float32)
		if logits.ndim == 1:
			logits = logits[np.newaxis, :]

		logits = logits - np.max(logits, axis=1, keepdims=True)
		exp_logits = np.exp(logits)
		probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

		results: list[dict] = []
		for row in probs:
			top_idx = int(np.argmax(row))
			label = self.LABELS[top_idx] if top_idx < len(self.LABELS) else str(top_idx)
			results.append(
				{
					"label": label,
					"confidence": float(row[top_idx]),
				}
			)
		return results

	def predict(self, face_images: Image.Image | list[Image.Image]) -> list[dict]:
		"""
		Output per face:
		  {
		    "label":      str,   # e.g. anger | contempt | disgust | fear | happy | neutral | sad | surprise"
		    "confidence": float  # 0.0 - 1.0
		  }
		Returns a list even for a single image input.
		"""
		if isinstance(face_images, Image.Image):
			face_images = [face_images]

		self._ensure_loaded()
		results: list[dict] = []

		for start in range(0, len(face_images), self.CHUNK_SIZE):
			chunk = face_images[start:start + self.CHUNK_SIZE]
			images = [
				np.asarray(face_image.convert("RGB"), dtype=np.uint8)
				for face_image in chunk
			]
			raw_outputs = self._infer(self.preprocess(images))
			results.extend(self.postprocess(raw_outputs))

		return results
