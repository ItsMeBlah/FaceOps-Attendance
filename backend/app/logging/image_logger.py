from __future__ import annotations

from datetime import date, timedelta
from io import BytesIO
import re
from typing import BinaryIO, Literal

from minio.deleteobjects import DeleteObject
from PIL import Image

from app.logging.minio import MinioStorage, minio_storage, normalize_user_name


ImageKind = Literal["raw", "aligned"]
LogDate = date | str | None


class ImageLogger:
    """Read and write raw/aligned face images in MinIO."""

    def __init__(self, storage: MinioStorage | None = None) -> None:
        self.storage = storage or minio_storage

    @property
    def logs_bucket(self) -> str:
        return self.storage.config.logs_bucket

    @property
    def aligned_images_bucket(self) -> str:
        return self.storage.config.aligned_images_bucket

    def create_user_paths(self, user_name: str, log_date: LogDate = None) -> None:
        self.storage.ensure_buckets()
        marker = b""
        for bucket_name in (self.logs_bucket, self.aligned_images_bucket):
            self.storage.client.put_object(
                bucket_name=bucket_name,
                object_name=f"{self.user_images_prefix(user_name, log_date)}/.keep",
                data=BytesIO(marker),
                length=len(marker),
                content_type="application/octet-stream",
            )

    def insert_raw_image(
        self,
        user_name: str,
        image: Image.Image | bytes | BinaryIO,
        index: int = 1,
        extension: str = "",
        log_date: LogDate = None,
    ) -> str:
        return self._put_face_image("raw", user_name, image, index, extension, log_date)

    def insert_aligned_image(
        self,
        user_name: str,
        image: Image.Image | bytes | BinaryIO,
        index: int = 1,
        extension: str = "",
        log_date: LogDate = None,
    ) -> str:
        return self._put_face_image("aligned", user_name, image, index, extension, log_date)

    def insert_face_images(
        self,
        user_name: str,
        raw_image: Image.Image | bytes | BinaryIO | None = None,
        aligned_image: Image.Image | bytes | BinaryIO | None = None,
        index: int = 1,
        extension: str = "",
        log_date: LogDate = None,
    ) -> dict[str, str]:
        object_names: dict[str, str] = {}
        if raw_image is not None:
            object_names["raw"] = self.insert_raw_image(
                user_name,
                raw_image,
                index,
                extension,
                log_date,
            )
        if aligned_image is not None:
            object_names["aligned"] = self.insert_aligned_image(
                user_name,
                aligned_image,
                index,
                extension,
                log_date,
            )
        return object_names

    def update_raw_image(
        self,
        user_name: str,
        image: Image.Image | bytes | BinaryIO,
        index: int = 1,
        extension: str = "",
        log_date: LogDate = None,
    ) -> str:
        return self.insert_raw_image(user_name, image, index, extension, log_date)

    def update_aligned_image(
        self,
        user_name: str,
        image: Image.Image | bytes | BinaryIO,
        index: int = 1,
        extension: str = "",
        log_date: LogDate = None,
    ) -> str:
        return self.insert_aligned_image(user_name, image, index, extension, log_date)

    def modify_image(
        self,
        kind: ImageKind,
        user_name: str,
        image: Image.Image | bytes | BinaryIO,
        index: int = 1,
        extension: str = "",
        log_date: LogDate = None,
    ) -> str:
        return self._put_face_image(kind, user_name, image, index, extension, log_date)

    def delete_raw_image(
        self,
        user_name: str,
        index: int = 1,
        extension: str = "",
        log_date: LogDate = None,
    ) -> None:
        self.delete_object(
            self.logs_bucket,
            self.raw_object_name(user_name, index, extension, log_date),
        )

    def delete_aligned_image(
        self,
        user_name: str,
        index: int = 1,
        extension: str = "",
        log_date: LogDate = None,
    ) -> None:
        self.delete_object(
            self.aligned_images_bucket,
            self.aligned_object_name(user_name, index, extension, log_date),
        )

    def delete_user_images(self, user_name: str, log_date: LogDate = None) -> None:
        self.storage.ensure_buckets()
        prefix = self.user_images_prefix(user_name, log_date) if log_date else None
        normalized_user = normalize_user_name(user_name)
        for bucket_name in (self.logs_bucket, self.aligned_images_bucket):
            objects = self.storage.client.list_objects(
                bucket_name,
                prefix=prefix,
                recursive=True,
            )
            objects_to_delete = (
                item
                for item in objects
                if prefix
                or self._object_belongs_to_user(item.object_name, normalized_user)
            )
            errors = self.storage.client.remove_objects(
                bucket_name,
                (DeleteObject(item.object_name) for item in objects_to_delete),
            )
            for error in errors:
                raise RuntimeError(f"Failed to delete MinIO object: {error}")

    def delete_object(self, bucket_name: str, object_name: str) -> None:
        self.storage.ensure_buckets()
        self.storage.client.remove_object(bucket_name, object_name)

    def list_user_images(
        self,
        user_name: str,
        bucket_name: str | None = None,
        log_date: LogDate = None,
    ) -> list[str]:
        self.storage.ensure_buckets()
        buckets = [bucket_name] if bucket_name else [self.logs_bucket, self.aligned_images_bucket]
        prefix = self.user_images_prefix(user_name, log_date) if log_date else None
        normalized_user = normalize_user_name(user_name)
        object_names: list[str] = []
        for bucket in buckets:
            object_names.extend(
                item.object_name
                for item in self.storage.client.list_objects(bucket, prefix=prefix, recursive=True)
                if not item.object_name.endswith("/.keep")
                and (prefix or self._object_belongs_to_user(item.object_name, normalized_user))
            )
        return object_names

    def next_image_index(self, kind: ImageKind, user_name: str, log_date: LogDate = None) -> int:
        bucket_name = self.logs_bucket if kind == "raw" else self.aligned_images_bucket
        stem = "raw_face" if kind == "raw" else "aligned_face"
        pattern = re.compile(rf"{re.escape(stem)}_(\d+)(?:\.[^/]+)?$")
        indexes = []
        for object_name in self.list_user_images(
            user_name,
            bucket_name=bucket_name,
            log_date=log_date or self.current_log_date(),
        ):
            match = pattern.search(object_name)
            if match:
                indexes.append(int(match.group(1)))
        return max(indexes, default=0) + 1

    def presigned_get_url(
        self,
        bucket_name: str,
        object_name: str,
        expires: timedelta = timedelta(hours=1),
    ) -> str:
        self.storage.ensure_buckets()
        return self.storage.client.presigned_get_object(bucket_name, object_name, expires=expires)

    def raw_object_name(
        self,
        user_name: str,
        index: int = 1,
        extension: str = "",
        log_date: LogDate = None,
    ) -> str:
        return self._object_name(user_name, "raw_face", index, extension, log_date)

    def aligned_object_name(
        self,
        user_name: str,
        index: int = 1,
        extension: str = "",
        log_date: LogDate = None,
    ) -> str:
        return self._object_name(user_name, "aligned_face", index, extension, log_date)

    def user_images_prefix(self, user_name: str, log_date: LogDate = None) -> str:
        return f"{self.normalize_log_date(log_date)}/{normalize_user_name(user_name)}/images"

    def current_log_date(self) -> str:
        return date.today().isoformat()

    def normalize_log_date(self, log_date: LogDate = None) -> str:
        if log_date is None:
            return self.current_log_date()
        if isinstance(log_date, date):
            return log_date.isoformat()
        cleaned = str(log_date).strip().replace("/", "-").replace("\\", "-")
        return cleaned or self.current_log_date()

    def _put_face_image(
        self,
        kind: ImageKind,
        user_name: str,
        image: Image.Image | bytes | BinaryIO,
        index: int,
        extension: str,
        log_date: LogDate,
    ) -> str:
        self.create_user_paths(user_name, log_date)
        if kind == "raw":
            bucket_name = self.logs_bucket
            object_name = self.raw_object_name(user_name, index, extension, log_date)
        elif kind == "aligned":
            bucket_name = self.aligned_images_bucket
            object_name = self.aligned_object_name(user_name, index, extension, log_date)
        else:
            raise ValueError("kind must be 'raw' or 'aligned'.")

        data, content_type = self._image_to_bytes(image, extension)
        self.storage.client.put_object(
            bucket_name=bucket_name,
            object_name=object_name,
            data=BytesIO(data),
            length=len(data),
            content_type=content_type,
        )
        return object_name

    def _object_name(
        self,
        user_name: str,
        stem: str,
        index: int,
        extension: str,
        log_date: LogDate,
    ) -> str:
        if index < 1:
            raise ValueError("Image index must be at least 1.")
        clean_extension = extension.strip().lower().lstrip(".")
        suffix = f".{clean_extension}" if clean_extension else ""
        return (
            f"{self.user_images_prefix(user_name, log_date)}/"
            f"{stem}_{index:03d}{suffix}"
        )

    @staticmethod
    def _object_belongs_to_user(object_name: str, normalized_user: str) -> bool:
        return (
            object_name.startswith(f"{normalized_user}/images/")
            or f"/{normalized_user}/images/" in object_name
        )

    def _image_to_bytes(
        self,
        image: Image.Image | bytes | BinaryIO,
        extension: str,
    ) -> tuple[bytes, str]:
        clean_extension = extension.strip().lower().lstrip(".")
        image_format = "png" if clean_extension == "png" else "jpg"
        content_type = "image/png" if image_format == "png" else "image/jpeg"

        if isinstance(image, bytes):
            return image, content_type

        if isinstance(image, Image.Image):
            output = BytesIO()
            save_format = "PNG" if image_format == "png" else "JPEG"
            image_to_save = image
            if save_format == "JPEG" and image.mode not in ("RGB", "L"):
                image_to_save = image.convert("RGB")
            image_to_save.save(output, format=save_format)
            return output.getvalue(), content_type

        if hasattr(image, "read"):
            return image.read(), content_type

        raise TypeError("image must be a PIL image, bytes, or file-like object.")
