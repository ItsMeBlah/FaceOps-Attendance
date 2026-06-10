from __future__ import annotations

from datetime import datetime, timedelta, timezone

from uuid6 import uuid7

from app.database import MongoDatabase, database


class ResultLogger:
	"""Store registered-user verification sessions and emotion totals."""

	EMOTIONS = (
		"anger",
		"contempt",
		"disgust",
		"fear",
		"happy",
		"neutral",
		"sad",
		"surprise",
	)

	def __init__(
		self,
		mongo: MongoDatabase | None = None,
		users_collection: str = "users",
		verification_collection: str = "verification_logs",
		emotion_collection: str = "emotion_stats",
		verification_session_timeout: timedelta = timedelta(minutes=2),
	) -> None:
		self.mongo = mongo or database
		self.users_collection = users_collection
		self.verification_collection = verification_collection
		self.emotion_collection = emotion_collection
		self.verification_session_timeout = verification_session_timeout
		self._indexes_ready = False

	def log_verification(
		self,
		user_id: str | None,
		user_name: str | None,
		matched: bool,
		confidence: float,
	) -> str | None:
		"""Create or update a verification presence session."""
		if not matched or not user_id or user_id.startswith("guest_"):
			return None

		self._ensure_indexes()
		now = self._utc_now()
		resolved_user_id = user_id
		resolved_user_name = self._resolve_user_name(
			resolved_user_id,
			user_name,
		)
		self._ensure_user(
			resolved_user_id,
			resolved_user_name,
			is_guest=False,
			timestamp=now,
		)

		latest_session = self.mongo.find_one(
			self.verification_collection,
			{"user_id": resolved_user_id},
			sort_by="last_seen",
			ascending=False,
		)
		if self._is_active_session(latest_session, now):
			self.mongo.update_one(
				self.verification_collection,
				{"_id": latest_session["_id"]},
				{
					"$set": {
						"user_name": resolved_user_name,
						"matched": bool(matched),
						"confidence": float(confidence),
						"last_seen": now,
					},
					"$max": {
						"highest_confidence": float(confidence),
					},
					"$inc": {
						"observation_count": 1,
					},
				},
			)
			return resolved_user_id

		log_id = str(uuid7())
		self.mongo.insert_one(
			self.verification_collection,
			{
				"_id": log_id,
				"user_id": resolved_user_id,
				"user_name": resolved_user_name,
				"matched": True,
				"confidence": float(confidence),
				"highest_confidence": float(confidence),
				"observation_count": 1,
				"first_seen": now,
				"last_seen": now,
			},
		)
		return resolved_user_id

	def log_emotion(
		self,
		user_id: str | None,
		emotion: str,
		confidence: float,
		user_name: str | None = None,
	) -> str | None:
		"""Increment one of the eight emotion counters for a user."""
		if not user_id or user_id.startswith("guest_"):
			return None

		self._ensure_indexes()
		timestamp = self._utc_now()
		resolved_user_id = user_id
		resolved_user_name = self._resolve_user_name(
			resolved_user_id,
			user_name,
		)
		normalized_emotion = self._normalize_emotion(emotion)

		self._ensure_user(
			resolved_user_id,
			resolved_user_name,
			is_guest=False,
			timestamp=timestamp,
		)
		self._ensure_emotion_document(
			resolved_user_id,
			resolved_user_name,
			timestamp,
		)
		self.mongo.update_one(
			self.emotion_collection,
			{"_id": resolved_user_id},
			{
				"$inc": {
					f"emotions.{normalized_emotion}": 1,
					"total": 1,
				},
				"$set": {
					"user_name": resolved_user_name,
					"last_emotion": normalized_emotion,
					"last_confidence": float(confidence),
					"updated_at": timestamp,
				},
			},
		)
		return resolved_user_id

	def upsert_registered_user(self, user_id: str, user_name: str) -> None:
		"""Create or update the MongoDB user that shares an ID with Qdrant."""
		if not user_id.strip():
			raise ValueError("User ID is required.")
		self._ensure_indexes()
		timestamp = self._utc_now()
		user_id = user_id.strip()
		user_name = user_name.strip() or user_id
		self._ensure_user(
			user_id,
			user_name,
			is_guest=False,
			timestamp=timestamp,
		)
		self._ensure_emotion_document(user_id, user_name, timestamp)

	def _ensure_user(
		self,
		user_id: str,
		user_name: str,
		is_guest: bool,
		timestamp: datetime,
	) -> None:
		self.mongo.update_one(
			self.users_collection,
			{"_id": user_id},
			{
				"$set": {
					"user_id": user_id,
					"user_name": user_name,
					"is_guest": is_guest,
					"updated_at": timestamp,
				},
				"$setOnInsert": {
					"created_at": timestamp,
				},
			},
			upsert=True,
		)

	def _ensure_emotion_document(
		self,
		user_id: str,
		user_name: str,
		timestamp: datetime,
	) -> None:
		self.mongo.update_one(
			self.emotion_collection,
			{"_id": user_id},
			{
				"$setOnInsert": {
					"user_id": user_id,
					"user_name": user_name,
					"emotions": {emotion: 0 for emotion in self.EMOTIONS},
					"total": 0,
					"created_at": timestamp,
					"updated_at": timestamp,
				},
			},
			upsert=True,
		)

	def _ensure_indexes(self) -> None:
		if self._indexes_ready:
			return
		verification_logs = self.mongo.collection(self.verification_collection)
		legacy_index = "user_id_1_timestamp_-1"
		if legacy_index in verification_logs.index_information():
			verification_logs.drop_index(legacy_index)
		verification_logs.create_index(
			[("user_id", 1), ("last_seen", -1)]
		)
		self.mongo.collection(self.emotion_collection).create_index(
			"user_id",
			unique=True,
		)
		self._indexes_ready = True

	def _is_active_session(
		self,
		session: dict | None,
		now: datetime,
	) -> bool:
		if not session or not isinstance(session.get("last_seen"), datetime):
			return False

		last_seen = session["last_seen"]
		if last_seen.tzinfo is None:
			last_seen = last_seen.replace(tzinfo=timezone.utc)
		return now - last_seen <= self.verification_session_timeout

	def _normalize_emotion(self, emotion: str) -> str:
		normalized = emotion.lower()
		if normalized not in self.EMOTIONS:
			raise ValueError(f"Unsupported emotion for logging: {emotion}")
		return normalized

	@staticmethod
	def _resolve_user_name(
		user_id: str,
		user_name: str | None,
	) -> str:
		if user_name and user_name != "unknown":
			return user_name
		return user_id

	@staticmethod
	def _utc_now() -> datetime:
		return datetime.now(timezone.utc)
