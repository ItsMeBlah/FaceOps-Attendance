"""
Simple MongoDB access:

	from app.database import database

	person_id = database.insert_one("people", {"name": "Minh", "active": True})
	person = database.find_one("people", {"name": "Minh"})
	people = database.find_many("people", {"active": True})
	database.update_one("people", {"name": "Minh"}, {"active": False})
	database.delete_one("people", {"name": "Minh"})
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from bson import ObjectId
from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

from app.config import settings


class MongoDatabase:
	"""Lazy MongoDB connection with small helpers for common CRUD queries."""

	def __init__(
		self,
		uri: str | None = None,
		database_name: str | None = None,
		timeout_ms: int | None = None,
	) -> None:
		self.uri = uri or settings.mongodb_uri
		self.database_name = database_name or settings.mongodb_database
		self.timeout_ms = timeout_ms or settings.mongodb_timeout_ms
		self._client: MongoClient | None = None
		self._database: Database | None = None

	@property
	def client(self) -> MongoClient:
		if self._client is None:
			self._client = MongoClient(
				self.uri,
				serverSelectionTimeoutMS=self.timeout_ms,
			)
		return self._client

	@property
	def db(self) -> Database:
		if self._database is None:
			self._database = self.client[self.database_name]
		return self._database

	def ping(self) -> bool:
		self.client.admin.command("ping")
		return True

	def collection(self, name: str) -> Collection:
		if not name.strip():
			raise ValueError("Collection name is required.")
		return self.db[name]

	def insert_one(self, collection: str, document: Mapping[str, Any]) -> str:
		result = self.collection(collection).insert_one(dict(document))
		return str(result.inserted_id)

	def find_one(
		self,
		collection: str,
		query: Mapping[str, Any] | None = None,
		sort_by: str | None = None,
		ascending: bool = True,
	) -> dict[str, Any] | None:
		if sort_by:
			direction = ASCENDING if ascending else DESCENDING
			document = self.collection(collection).find_one(
				dict(query or {}),
				sort=[(sort_by, direction)],
			)
		else:
			document = self.collection(collection).find_one(dict(query or {}))

		return self._serialize(document) if document is not None else None

	def find_many(
		self,
		collection: str,
		query: Mapping[str, Any] | None = None,
		limit: int = 100,
		sort_by: str = "_id",
		ascending: bool = True,
	) -> list[dict[str, Any]]:
		if limit < 1:
			raise ValueError("Limit must be at least 1.")

		direction = ASCENDING if ascending else DESCENDING
		cursor = (
			self.collection(collection)
			.find(dict(query or {}))
			.sort(sort_by, direction)
			.limit(limit)
		)
		return [self._serialize(document) for document in cursor]

	def update_one(
		self,
		collection: str,
		query: Mapping[str, Any],
		updates: Mapping[str, Any],
		upsert: bool = False,
	) -> int:
		update_document = dict(updates)
		if not any(key.startswith("$") for key in update_document):
			update_document = {"$set": update_document}

		result = self.collection(collection).update_one(
			dict(query),
			update_document,
			upsert=upsert,
		)
		return result.modified_count

	def delete_one(self, collection: str, query: Mapping[str, Any]) -> int:
		result = self.collection(collection).delete_one(dict(query))
		return result.deleted_count

	def count(self, collection: str, query: Mapping[str, Any] | None = None) -> int:
		return self.collection(collection).count_documents(dict(query or {}))

	def close(self) -> None:
		if self._client is not None:
			self._client.close()
		self._client = None
		self._database = None

	@classmethod
	def _serialize(cls, value: Any) -> Any:
		if isinstance(value, ObjectId):
			return str(value)
		if isinstance(value, dict):
			return {key: cls._serialize(item) for key, item in value.items()}
		if isinstance(value, list):
			return [cls._serialize(item) for item in value]
		return value


database = MongoDatabase()
