from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import onnxruntime


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_ALIGN_DIR = PROJECT_ROOT / "test_align"
DEFAULT_INPUT_ROOT = (
    PROJECT_ROOT
    / "dataset"
    / "11-785-fall-20-homework-2-part-2"
    / "classification_data"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "dataset"
    / "11-785-fall-20-homework-2-part-2"
    / "classification_data_aligned"
)
DEFAULT_DET_MODEL = (
    Path("/home/minhcao/Swinburne/COS30082/CustomProject/-Facial-Recognition-with-Emotion-and-Liveness")
    / "training_module"
    / "Face_detection_module"
    / "model"
    / "buffalo_l"
    / "det_10g.onnx"
)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train_data", "val_data", "test_data")

onnxruntime.set_default_logger_severity(3)

if str(TEST_ALIGN_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_ALIGN_DIR))

from face_align import norm_crop  # noqa: E402
from scrfd import SCRFD  # noqa: E402

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


def iter_images(input_root: Path, splits: Iterable[str]) -> Iterable[Path]:
    for split in splits:
        split_dir = input_root / split
        if not split_dir.is_dir():
            print(f"warning: split not found, skipping: {split_dir}")
            continue
        for path in sorted(split_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                yield path


def progress(items: list[Path], enabled: bool):
    if enabled and tqdm is not None:
        return tqdm(items, desc="aligning", dynamic_ncols=True)
    return items


def choose_highest_confidence_face(bboxes: np.ndarray, kpss: np.ndarray) -> np.ndarray | None:
    if bboxes is None or kpss is None or bboxes.shape[0] == 0:
        return None
    best_index = int(np.argmax(bboxes[:, 4]))
    return kpss[best_index]


def align_image(detector: SCRFD, image_path: Path, image_size: int) -> np.ndarray | None:
    image = cv2.imread(str(image_path))
    if image is None:
        return None

    bboxes, kpss = detector.autodetect(image, max_num=0)
    landmarks = choose_highest_confidence_face(bboxes, kpss)
    if landmarks is None:
        return None

    return norm_crop(image, landmarks, image_size=image_size, mode="arcface")


def write_failure_log(path: Path, rows: list[tuple[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "reason"])
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Align classification_data to ArcFace 128x128 crops with SCRFD landmarks.")
    parser.add_argument("--input-root", default=str(DEFAULT_INPUT_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--det-model", default=str(DEFAULT_DET_MODEL))
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--ctx-id", type=int, default=0, help="0 for CUDA, -1 for CPU when supported by the detector wrapper.")
    parser.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS)
    parser.add_argument("--quality", type=int, default=95)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Optional debug limit. 0 means process all images.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    det_model = Path(args.det_model).expanduser().resolve()

    if not input_root.is_dir():
        raise FileNotFoundError(f"input root not found: {input_root}")
    if not det_model.is_file():
        raise FileNotFoundError(f"SCRFD detector model not found: {det_model}")

    detector = SCRFD(str(det_model))
    detector.prepare(args.ctx_id)

    image_paths = list(iter_images(input_root, args.splits))
    if args.limit > 0:
        image_paths = image_paths[: args.limit]

    output_root.mkdir(parents=True, exist_ok=True)
    failures: list[tuple[str, str]] = []
    processed = 0
    skipped = 0

    for image_path in progress(image_paths, enabled=not args.no_progress):
        relative_path = image_path.relative_to(input_root)
        output_path = output_root / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.exists() and not args.overwrite:
            skipped += 1
            continue

        aligned = align_image(detector, image_path, image_size=args.image_size)
        if aligned is None:
            failures.append((str(relative_path), "read_error_or_face_not_found"))
            if args.fail_fast:
                raise RuntimeError(f"failed to align {image_path}")
            continue

        ok = cv2.imwrite(str(output_path), aligned, [int(cv2.IMWRITE_JPEG_QUALITY), args.quality])
        if not ok:
            failures.append((str(relative_path), "write_failed"))
            if args.fail_fast:
                raise RuntimeError(f"failed to write {output_path}")
            continue

        processed += 1

    failure_log = output_root / "alignment_failures.csv"
    write_failure_log(failure_log, failures)
    print(f"input_root: {input_root}")
    print(f"output_root: {output_root}")
    print(f"processed: {processed}")
    print(f"skipped_existing: {skipped}")
    print(f"failed: {len(failures)}")
    print(f"failure_log: {failure_log}")


if __name__ == "__main__":
    main()
