from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from minio import Minio


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExtractedObject:
    bucket: str
    object_name: str
    date: str
    user_name: str
    local_path: str
    size: int | None = None


def extract_aligned_images(
    config: dict[str, Any],
    start_date: date,
    end_date: date,
    raw_dir: Path,
) -> list[ExtractedObject]:
    """Download aligned face images from MinIO for an inclusive date range."""
    minio_config = config.get("minio", {})
    bucket_name = str(minio_config.get("bucket", "aligned-images"))
    valid_extensions = {
        str(extension).lower()
        for extension in minio_config.get("valid_extensions", [".jpg", ".jpeg", ".png"])
    }
    default_extension = str(minio_config.get("default_extension", ".jpg"))

    if config.get("pipeline", {}).get("overwrite", True) and raw_dir.exists():
        _remove_directory(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    client = Minio(
        endpoint=str(minio_config.get("endpoint", "localhost:9000")),
        access_key=str(minio_config.get("access_key", "faceguard")),
        secret_key=str(minio_config.get("secret_key", "faceguardsecret")),
        secure=bool(minio_config.get("secure", False)),
    )

    downloaded: list[ExtractedObject] = []
    for current_date in iter_dates(start_date, end_date):
        date_key = current_date.isoformat()
        prefix = f"{date_key}/"
        logger.info("Listing MinIO objects bucket=%s prefix=%s", bucket_name, prefix)
        objects = client.list_objects(bucket_name, prefix=prefix, recursive=True)

        for item in objects:
            object_name = item.object_name
            if not object_name or object_name.endswith("/.keep") or object_name.endswith("/"):
                continue

            parsed = parse_aligned_object_name(object_name, date_key)
            if parsed is None:
                continue

            user_name, source_file_name = parsed
            local_file_name = normalized_local_filename(
                date_key=date_key,
                source_file_name=source_file_name,
                default_extension=default_extension,
            )
            if Path(local_file_name).suffix.lower() not in valid_extensions:
                continue

            user_dir = raw_dir / safe_path_segment(user_name)
            user_dir.mkdir(parents=True, exist_ok=True)
            local_path = unique_path(user_dir / local_file_name)

            client.fget_object(bucket_name, object_name, str(local_path))
            downloaded.append(
                ExtractedObject(
                    bucket=bucket_name,
                    object_name=object_name,
                    date=date_key,
                    user_name=user_name,
                    local_path=str(local_path),
                    size=getattr(item, "size", None),
                )
            )

    write_extraction_manifest(raw_dir, start_date, end_date, downloaded)
    logger.info("Downloaded %s aligned images into %s", len(downloaded), raw_dir)
    return downloaded


def parse_aligned_object_name(object_name: str, date_key: str) -> tuple[str, str] | None:
    prefix = f"{date_key}/"
    if not object_name.startswith(prefix):
        return None

    remainder = object_name[len(prefix):]
    marker = "/images/"
    if marker not in remainder:
        return None

    user_name, file_name = remainder.split(marker, 1)
    user_name = user_name.strip()
    file_name = file_name.strip()
    if not user_name or not file_name:
        return None
    return user_name, Path(file_name).name


def iter_dates(start_date: date, end_date: date):
    if end_date < start_date:
        raise ValueError("end_date must be greater than or equal to start_date.")

    current_date = start_date
    while current_date <= end_date:
        yield current_date
        current_date += timedelta(days=1)


def normalized_local_filename(
    date_key: str,
    source_file_name: str,
    default_extension: str,
) -> str:
    source_path = Path(source_file_name)
    stem = safe_path_segment(source_path.stem or "image")
    suffix = source_path.suffix.lower() or default_extension
    if not suffix.startswith("."):
        suffix = f".{suffix}"
    return f"{date_key}_{stem}{suffix}"


def safe_path_segment(value: str) -> str:
    cleaned = value.strip().replace("/", "_").replace("\\", "_")
    return cleaned or "unknown"


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path

    for index in range(2, 100_000):
        candidate = path.with_name(f"{path.stem}_{index}{path.suffix}")
        if not candidate.exists():
            return candidate

    raise RuntimeError(f"Could not create a unique path for {path}")


def write_extraction_manifest(
    raw_dir: Path,
    start_date: date,
    end_date: date,
    objects: list[ExtractedObject],
) -> Path:
    manifest_path = raw_dir / "extraction_manifest.json"
    payload = {
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "count": len(objects),
        "objects": [asdict(item) for item in objects],
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return manifest_path


def _remove_directory(path: Path) -> None:
    import shutil

    shutil.rmtree(path)
