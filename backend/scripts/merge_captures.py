"""
Filter pipeline captures into training dataset folders by predicted label.

Workflow:
  1. Load merge_config.yaml (paths, thresholds, enabled targets).
  2. Scan captures_dir for {stamp}_frame.jpg + {stamp}_predictions.json.
  3. For each face in the JSON:
     - Crop using the normalized bbox.
     - For each enabled target, read the predicted label + confidence.
     - If confidence >= min_confidence, copy the crop into paths from merge_config.yaml:
         emotion:  <root>/<split>/<emotion>/capture_...jpg
         liveness: <root>/<LCC_split>/real|spoof/capture_...jpg
         recognition: <root>/train_data/<person>/capture_...jpg

Examples:
  - anti_spoofing spoof @ 0.92, split training -> <LCC_FASD>/LCC_FASD_training/spoof/...
  - emotion happy @ 0.88, split train -> <fer2013>/train/happy/...
  - below threshold                 -> skipped (not appended)

Run from repo root:
  python backend/scripts/merge_captures.py
  python backend/scripts/merge_captures.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_CONFIG = SCRIPT_DIR / "merge_config.yaml"
DEFAULT_MANIFEST = SCRIPT_DIR / ".merge_manifest.json"

# Face recognition (classification_data)
FACE_RECOGNITION_SPLITS = {"train": "train_data", "val": "val_data", "test": "test_data"}

# LCC_FASD: root points at the LCC_FASD directory (parent of LCC_FASD_* folders)
LCC_FASD_SPLITS = {
    "train": "LCC_FASD_training",
    "training": "LCC_FASD_training",
    "development": "LCC_FASD_development",
    "dev": "LCC_FASD_development",
    "evaluation": "LCC_FASD_evaluation",
    "eval": "LCC_FASD_evaluation",
    "test": "LCC_FASD_evaluation",
}

# FER2013: root points at fer2013 (parent of train/ and test/)
FER_EMOTION_SPLITS = {"train": "train", "test": "test"}

# AffectNet: root points at AffectNet (parent of Train/ and Test/)
AFFECTNET_EMOTION_SPLITS = {"train": "Train", "test": "Test"}

# Backend / AffectNet label -> FER2013 folder name (None = never append for FER)
EMOTION_FER_MAP: dict[str, str | None] = {
    "anger": "angry",
    "contempt": None,
    "disgust": "disgust",
    "fear": "fear",
    "happy": "happy",
    "neutral": "neutral",
    "sad": "sad",
    "surprise": "surprise",
    "angry": "angry",
}

LABEL_FROM_TO_JSON = {
    "emotion": ("emotion", "label", "confidence"),
    "liveness": ("anti_spoofing", "label", "confidence"),
    "recognition": ("recognition", "label", "confidence"),
}


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return data


def resolve_repo_root(config: dict[str, Any]) -> Path:
    raw = (config.get("repo_root") or "").strip()
    if raw in ("", "."):
        return DEFAULT_REPO_ROOT
    path = Path(raw).expanduser()
    return path.resolve() if path.is_absolute() else (DEFAULT_REPO_ROOT / path).resolve()


def resolve_path(repo_root: Path, raw: str, field_name: str) -> Path:
    value = (raw or "").strip()
    if not value:
        raise ValueError(f"Missing required path: {field_name}")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def threshold_for_target(config: dict[str, Any], target: dict[str, Any]) -> float:
    if "min_confidence" in target and target["min_confidence"] is not None:
        return float(target["min_confidence"])
    return float(config.get("min_confidence", 0.0))


def load_manifest(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    items = data.get("ingested", []) if isinstance(data, dict) else []
    return {str(item) for item in items}


def save_manifest(path: Path, ingested: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"ingested": sorted(ingested)}, indent=2), encoding="utf-8")


def sanitize_label(label: str) -> str:
    cleaned = re.sub(r"[^\w\-.]+", "_", label.strip().lower())
    cleaned = cleaned.strip("._")
    if not cleaned:
        raise ValueError("empty label after sanitization")
    return cleaned


def crop_face(image: Image.Image, face: dict[str, Any]) -> Image.Image:
    box = face["face"]["bbox"]
    width, height = image.size
    left = int(box["x"] * width)
    top = int(box["y"] * height)
    right = int((box["x"] + box["w"]) * width)
    bottom = int((box["y"] + box["h"]) * height)
    left = max(0, min(left, width - 1))
    top = max(0, min(top, height - 1))
    right = max(left + 1, min(right, width))
    bottom = max(top + 1, min(bottom, height))
    return image.crop((left, top, right, bottom))


def read_prediction(target_name: str, target: dict[str, Any], face: dict[str, Any]) -> Prediction | None:
    source = str(target.get("label_from", "emotion")).strip().lower()
    keys = LABEL_FROM_TO_JSON.get(source)
    if not keys:
        raise ValueError(f"Unknown label_from={source!r} for target {target_name}")

    section_key, label_key, confidence_key = keys
    section = face.get(section_key) or {}
    raw_label = section.get(label_key, "")
    raw_confidence = section.get(confidence_key, 0.0)

    try:
        confidence = float(raw_confidence)
    except (TypeError, ValueError):
        confidence = 0.0

    if source == "recognition":
        if target.get("only_matched") and not face.get("recognition", {}).get("matched"):
            return None
        if not raw_label or str(raw_label).lower() == "unknown":
            return None

    if source == "emotion" and target_name == "emotion_fer":
        mapped = EMOTION_FER_MAP.get(str(raw_label), str(raw_label))
        if mapped is None:
            return None
        return Prediction(label=sanitize_label(mapped), confidence=confidence)

    if not raw_label:
        return None

    return Prediction(label=sanitize_label(str(raw_label)), confidence=confidence)


def _split_subdirectory(target_name: str, target: dict[str, Any]) -> str:
    """Resolve config split -> on-disk folder name for this target."""
    if target.get("split_dir"):
        return str(target["split_dir"]).strip()

    split = str(target.get("split", "train")).strip().lower()

    if target_name == "face_recognition":
        subdir = FACE_RECOGNITION_SPLITS.get(split)
        if not subdir:
            raise ValueError(
                f"targets.{target_name}.split must be one of {sorted(FACE_RECOGNITION_SPLITS)}"
            )
        return subdir

    if target_name == "anti_spoofing":
        subdir = LCC_FASD_SPLITS.get(split)
        if not subdir:
            raise ValueError(
                f"targets.{target_name}.split must be one of: "
                f"training, development, evaluation (aliases: train, dev, test, ...)"
            )
        return subdir

    if target_name == "emotion_fer":
        subdir = FER_EMOTION_SPLITS.get(split)
        if not subdir:
            raise ValueError(
                f"targets.{target_name}.split must be one of {sorted(FER_EMOTION_SPLITS)}"
            )
        return subdir

    if target_name == "emotion_affectnet":
        subdir = AFFECTNET_EMOTION_SPLITS.get(split)
        if not subdir:
            raise ValueError(
                f"targets.{target_name}.split must be one of {sorted(AFFECTNET_EMOTION_SPLITS)}"
            )
        return subdir

    raise ValueError(f"Unknown target {target_name!r} — cannot resolve split folder")


def destination_dir(
    target_name: str,
    target: dict[str, Any],
    repo_root: Path,
    class_label: str,
) -> Path:
    """
    All paths come from merge_config.yaml `root` + `split` (+ class label).

    Layouts:
      emotion:       <root>/<train|test>/<emotion>/
      anti_spoofing: <root>/LCC_FASD_<split>/real|spoof/
      face_recognition: <root>/train_data/<person>/
    """
    root = resolve_path(repo_root, str(target.get("root", "")), f"targets.{target_name}.root")
    split_dir = _split_subdirectory(target_name, target)
    return root / split_dir / class_label


def list_capture_stamps(captures_dir: Path) -> list[str]:
    stamps: list[str] = []
    for json_path in sorted(captures_dir.glob("*_predictions.json")):
        stamp = json_path.name[: -len("_predictions.json")]
        if (captures_dir / f"{stamp}_frame.jpg").is_file():
            stamps.append(stamp)
    return stamps


def merge_capture(
    stamp: str,
    captures_dir: Path,
    enabled_targets: list[tuple[str, dict[str, Any]]],
    repo_root: Path,
    config: dict[str, Any],
    *,
    dry_run: bool,
) -> tuple[int, list[str]]:
    json_path = captures_dir / f"{stamp}_predictions.json"
    frame_path = captures_dir / f"{stamp}_frame.jpg"

    with json_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    faces = payload.get("faces", [])
    if not faces:
        return 0, ["no faces in JSON"]

    min_detection = float(config.get("min_detection_confidence", 0.0))
    image = Image.open(frame_path).convert("RGB")
    written = 0
    notes: list[str] = []

    for face_index, face in enumerate(faces):
        detection_conf = float(face.get("face", {}).get("detection_confidence", 0.0))
        if detection_conf < min_detection:
            notes.append(
                f"face {face_index}: detection confidence {detection_conf:.3f} < {min_detection:.3f}"
            )
            continue

        try:
            crop = crop_face(image, face)
        except (KeyError, TypeError, ValueError) as exc:
            notes.append(f"face {face_index}: bad bbox ({exc})")
            continue

        for target_name, target in enabled_targets:
            min_conf = threshold_for_target(config, target)

            try:
                prediction = read_prediction(target_name, target, face)
            except ValueError as exc:
                notes.append(f"face {face_index} / {target_name}: {exc}")
                continue

            if prediction is None:
                continue

            if prediction.confidence < min_conf:
                notes.append(
                    f"face {face_index} / {target_name}: "
                    f"{prediction.label} {prediction.confidence:.3f} < {min_conf:.3f}"
                )
                continue

            out_dir = destination_dir(target_name, target, repo_root, prediction.label)
            out_path = out_dir / f"capture_{stamp}_face{face_index}.jpg"

            if dry_run:
                print(
                    f"[dry-run] {prediction.label} ({prediction.confidence:.3f}) "
                    f"-> {out_path}"
                )
            else:
                out_dir.mkdir(parents=True, exist_ok=True)
                crop.save(out_path, format="JPEG", quality=95)
            written += 1

    return written, notes


def collect_enabled_targets(config: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    targets = config.get("targets") or {}
    if not isinstance(targets, dict):
        raise ValueError("config.targets must be a mapping")

    enabled: list[tuple[str, dict[str, Any]]] = []
    for name, target in targets.items():
        if isinstance(target, dict) and target.get("enabled"):
            enabled.append((name, target))
    return enabled


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Filter captures into training folders by predicted label and confidence.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    config = load_config(args.config.resolve())
    repo_root = resolve_repo_root(config)

    captures_raw = (config.get("captures_dir") or "").strip()
    if not captures_raw:
        print("Set captures_dir in merge_config.yaml.", file=sys.stderr)
        return 1
    captures_dir = resolve_path(repo_root, captures_raw, "captures_dir")

    manifest_raw = (config.get("manifest_path") or "").strip()
    manifest_path = (
        resolve_path(repo_root, manifest_raw, "manifest_path")
        if manifest_raw
        else DEFAULT_MANIFEST
    )

    enabled_targets = collect_enabled_targets(config)
    if not enabled_targets:
        print("No targets enabled in merge_config.yaml.", file=sys.stderr)
        return 1

    if not captures_dir.is_dir():
        print(f"Captures directory not found: {captures_dir}", file=sys.stderr)
        return 1

    ingested = load_manifest(manifest_path)
    stamps = list_capture_stamps(captures_dir)
    pending = [s for s in stamps if s not in ingested]

    print(f"Repo root:        {repo_root}")
    print(f"Captures:         {captures_dir} ({len(stamps)} pairs, {len(pending)} new)")
    print(f"Manifest:         {manifest_path}")
    print(f"Default min conf: {float(config.get('min_confidence', 0.0)):.3f}")
    print(f"Enabled targets:  {', '.join(n for n, _ in enabled_targets)}")
    if args.dry_run:
        print("Mode:             dry-run")

    total_written = 0
    merged_stamps: list[str] = []

    for stamp in pending:
        try:
            count, notes = merge_capture(
                stamp,
                captures_dir,
                enabled_targets,
                repo_root,
                config,
                dry_run=args.dry_run,
            )
        except Exception as exc:
            print(f"[skip] {stamp}: {exc}", file=sys.stderr)
            continue

        for note in notes:
            print(f"  {stamp}: {note}")

        if count > 0:
            total_written += count
            merged_stamps.append(stamp)
            if not args.dry_run:
                ingested.add(stamp)
        else:
            print(f"[skip] {stamp}: nothing written (all below threshold or unroutable)")

    if not args.dry_run and merged_stamps:
        save_manifest(manifest_path, ingested)

    print(f"Done. Appended {total_written} image(s) from {len(merged_stamps)} capture(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
