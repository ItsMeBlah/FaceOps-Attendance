from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

import urllib3
from minio import Minio

from app.config import settings


@dataclass(frozen=True)
class MinioConfig:
    endpoint: str = settings.minio_endpoint
    access_key: str = settings.minio_access_key
    secret_key: str = settings.minio_secret_key
    secure: bool = settings.minio_secure
    region: str = settings.minio_region
    logs_bucket: str = settings.minio_logs_bucket
    aligned_images_bucket: str = settings.minio_aligned_images_bucket
    timeout_seconds: float = settings.minio_timeout_seconds


class MinioStorage:
    """Initialize MinIO access and keep required FaceGuard buckets ready."""

    def __init__(self, config: MinioConfig | None = None) -> None:
        self.config = config or MinioConfig()
        self._client: Minio | None = None

    @property
    def client(self) -> Minio:
        if self._client is None:
            http_client = urllib3.PoolManager(
                timeout=urllib3.Timeout(
                    connect=self.config.timeout_seconds,
                    read=self.config.timeout_seconds,
                ),
                retries=urllib3.Retry(total=0),
            )
            self._client = Minio(
                endpoint=self.config.endpoint,
                access_key=self.config.access_key,
                secret_key=self.config.secret_key,
                secure=self.config.secure,
                region=self.config.region,
                http_client=http_client,
            )
        return self._client

    @property
    def buckets(self) -> tuple[str, str]:
        return (self.config.logs_bucket, self.config.aligned_images_bucket)

    def ensure_buckets(self) -> None:
        """Create the logs and aligned-images buckets when they do not exist."""
        for bucket_name in self.buckets:
            if not self.client.bucket_exists(bucket_name):
                self.client.make_bucket(bucket_name, location=self.config.region)

    def ensure_user_paths(self, user_name: str) -> None:
        """
        Create marker objects for the expected user image prefixes.

        MinIO/S3 uses object prefixes instead of real folders, so the path
        ``{user_name}/images`` exists once an object is written under it.
        """
        self.ensure_buckets()
        marker = b""
        for bucket_name in self.buckets:
            self.client.put_object(
                bucket_name=bucket_name,
                object_name=f"{normalize_user_name(user_name)}/images/.keep",
                data=BytesIO(marker),
                length=len(marker),
                content_type="application/octet-stream",
            )


def normalize_user_name(user_name: str) -> str:
    """Return a safe prefix segment while keeping the display name readable."""
    cleaned = user_name.strip()
    cleaned = cleaned.replace("/", "_").replace("\\", "_")
    return cleaned or "unknown"


minio_storage = MinioStorage()
