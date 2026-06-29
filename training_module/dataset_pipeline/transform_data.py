from __future__ import annotations

import json
import logging
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TransformReport:
    input_count: int
    accepted_count: int
    rejected_count: int
    rejected: list[dict[str, str]]


def transform_dataset(
    config: dict[str, Any],
    raw_dir: Path,
    dataset_dir: Path,
) -> TransformReport:
    """Filter extracted images and write the final class-folder dataset."""
    transform_config = config.get("transform", {})
    min_width = int(transform_config.get("min_width", 64))
    min_height = int(transform_config.get("min_height", 64))
    blur_config = transform_config.get("blur", {})
    blur_enabled = bool(blur_config.get("enabled", True))
    blur_threshold = float(blur_config.get("threshold", 80.0))

    if config.get("pipeline", {}).get("overwrite", True) and dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    remove_dataset_metadata_files(dataset_dir)

    input_images = [
        path
        for path in raw_dir.rglob("*")
        if path.is_file() and path.name not in {"extraction_manifest.json", "transform_report.json"}
    ]
    rejected: list[dict[str, str]] = []
    accepted_count = 0

    for source_path in input_images:
        relative_path = source_path.relative_to(raw_dir)
        target_path = dataset_dir / relative_path

        reason = rejection_reason(
            source_path=source_path,
            min_width=min_width,
            min_height=min_height,
            blur_enabled=blur_enabled,
            blur_threshold=blur_threshold,
        )
        if reason is not None:
            rejected.append({"path": str(source_path), "reason": reason})
            continue

        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)
        accepted_count += 1

    report = TransformReport(
        input_count=len(input_images),
        accepted_count=accepted_count,
        rejected_count=len(rejected),
        rejected=rejected,
    )
    write_transform_report(raw_dir, report)

    logger.info(
        "Transformed dataset raw=%s output=%s accepted=%s rejected=%s",
        raw_dir,
        dataset_dir,
        report.accepted_count,
        report.rejected_count,
    )
    return report


def rejection_reason(
    source_path: Path,
    min_width: int,
    min_height: int,
    blur_enabled: bool,
    blur_threshold: float,
) -> str | None:
    try:
        with Image.open(source_path) as image:
            width, height = image.size
            if width < min_width or height < min_height:
                return f"too_small:{width}x{height}"
    except (UnidentifiedImageError, OSError) as exc:
        return f"unreadable:{exc}"

    if blur_enabled:
        blur_score = image_blur_score(source_path)
        if blur_score < blur_threshold:
            return f"too_blurry:{blur_score:.3f}"

    return None


def image_blur_score(source_path: Path) -> float:
    try:
        import cv2
        import numpy as np
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Blur filtering requires opencv-python-headless and numpy. "
            "Disable transform.blur.enabled or install those packages."
        ) from exc

    with Image.open(source_path) as image:
        grayscale = image.convert("L")
        pixels = np.asarray(grayscale)
    return float(cv2.Laplacian(pixels, cv2.CV_64F).var())


def write_transform_report(report_dir: Path, report: TransformReport) -> Path:
    report_path = report_dir / "transform_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(asdict(report), handle, indent=2)
    return report_path


def remove_dataset_metadata_files(dataset_dir: Path) -> None:
    for file_name in ("extraction_manifest.json", "transform_report.json"):
        metadata_path = dataset_dir / file_name
        if metadata_path.is_file():
            metadata_path.unlink()
