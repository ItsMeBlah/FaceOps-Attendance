from __future__ import annotations

import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
from qdrant_client import QdrantClient, models

from app.config import settings


@dataclass(frozen=True)
class VectorMatch:
	employee_id: str
	employee_name: str
	score: float


class VectorStoreService:
	def __init__(
		self,
		url: str | None = None,
		api_key: str | None = None,
		collection_name: str | None = None,
		vector_size: int | None = None,
		top_k: int | None = None,
	) -> None:
		self.url = url or settings.qdrant_url
		self.api_key = api_key if api_key is not None else settings.qdrant_api_key
		self.collection_name = collection_name or settings.qdrant_collection
		self.vector_size = vector_size or settings.qdrant_vector_size
		self.top_k = top_k or settings.qdrant_top_k

		self._client: QdrantClient | None = None
		self._collection_ready = False

	@property
	def client(self) -> QdrantClient:
		if self._client is None:
			self._client = QdrantClient(
				url=self.url,
				api_key=self.api_key or None,
			)
		return self._client

	def register_embeddings(
		self,
		employee_id: str,
		employee_name: str,
		embeddings: list[np.ndarray],
	) -> None:
		self._ensure_collection()
		self.delete_employee(employee_id)

		points = []
		for index, embedding in enumerate(embeddings):
			vector = self._to_vector(embedding)
			points.append(
				models.PointStruct(
					id=str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{employee_id}:{index}")),
					vector=vector,
					payload={
						"employee_id": employee_id,
						"employee_name": employee_name,
						"image_index": index,
					},
				)
			)

		if points:
			self.client.upsert(
				collection_name=self.collection_name,
				points=points,
				wait=True,
			)

	def search(self, embedding: np.ndarray, top_k: int | None = None) -> list[VectorMatch]:
		self._ensure_collection()
		vector = self._to_vector(embedding)
		limit = top_k or self.top_k

		points = self._query_points(vector, limit)
		matches: list[VectorMatch] = []
		for point in points:
			payload = point.payload or {}
			employee_id = str(payload.get("employee_id", ""))
			if not employee_id:
				continue
			matches.append(
				VectorMatch(
					employee_id=employee_id,
					employee_name=str(payload.get("employee_name") or employee_id),
					score=float(point.score),
				)
			)
		return matches

	def pick_majority_match(self, embedding: np.ndarray) -> VectorMatch | None:
		matches = self.search(embedding, top_k=self.top_k)
		if not matches:
			return None

		counts = Counter(match.employee_id for match in matches)
		scores_by_id: dict[str, list[float]] = defaultdict(list)
		names_by_id: dict[str, str] = {}
		for match in matches:
			scores_by_id[match.employee_id].append(match.score)
			names_by_id[match.employee_id] = match.employee_name

		winner_id = max(
			counts,
			key=lambda employee_id: (
				counts[employee_id],
				sum(scores_by_id[employee_id]) / len(scores_by_id[employee_id]),
				max(scores_by_id[employee_id]),
			),
		)

		return VectorMatch(
			employee_id=winner_id,
			employee_name=names_by_id.get(winner_id, winner_id),
			score=max(scores_by_id[winner_id]),
		)

	def delete_employee(self, employee_id: str) -> None:
		self._ensure_collection()
		self.client.delete(
			collection_name=self.collection_name,
			points_selector=models.FilterSelector(
				filter=models.Filter(
					must=[
						models.FieldCondition(
							key="employee_id",
							match=models.MatchValue(value=employee_id),
						)
					]
				)
			),
			wait=True,
		)

	def employee_exists(self, employee_id: str) -> bool:
		self._ensure_collection()
		result = self.client.count(
			collection_name=self.collection_name,
			count_filter=models.Filter(
				must=[
					models.FieldCondition(
						key="employee_id",
						match=models.MatchValue(value=employee_id),
					)
				]
			),
			exact=True,
		)
		return result.count > 0

	def _ensure_collection(self) -> None:
		if self._collection_ready:
			return

		if not self.client.collection_exists(collection_name=self.collection_name):
			self.client.create_collection(
				collection_name=self.collection_name,
				vectors_config=models.VectorParams(
					size=self.vector_size,
					distance=models.Distance.COSINE,
				),
			)

		self._collection_ready = True

	def _query_points(self, vector: list[float], limit: int) -> list[Any]:
		try:
			response = self.client.query_points(
				collection_name=self.collection_name,
				query=vector,
				limit=limit,
				with_payload=True,
			)
			return list(response.points)
		except AttributeError:
			return list(
				self.client.search(
					collection_name=self.collection_name,
					query_vector=vector,
					limit=limit,
					with_payload=True,
				)
			)

	def _to_vector(self, embedding: np.ndarray) -> list[float]:
		vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
		if vector.size != self.vector_size:
			raise ValueError(
				f"Embedding size must be {self.vector_size}, got {vector.size}."
			)
		return vector.tolist()
