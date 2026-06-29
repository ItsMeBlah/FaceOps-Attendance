from __future__ import annotations

import argparse
import logging
from datetime import date
from pathlib import Path
from typing import Any

import yaml

try:
    from .dvc_logging import log_dataset_to_dvc
    from .extract_data_minio import extract_aligned_images
    from .transform_data import transform_dataset
except ImportError:
    from dvc_logging import log_dataset_to_dvc
    from extract_data_minio import extract_aligned_images
    from transform_data import transform_dataset


PIPELINE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PIPELINE_ROOT.parents[1]
DEFAULT_CONFIG_PATH = PIPELINE_ROOT / "config.yaml"


def main() -> None:
    configure_logging()
    args = parse_args()
    config = load_config(args.config)
    start_date = parse_or_prompt_date(args.start_date, "Start date (YYYY-MM-DD): ")
    end_date = parse_or_prompt_date(args.end_date, "End date (YYYY-MM-DD): ")

    if end_date < start_date:
        raise ValueError("End date must be greater than or equal to start date.")

    if args.no_dvc:
        config.setdefault("dvc", {})["enabled"] = False
    if args.no_dvc_push:
        config.setdefault("dvc", {})["push"] = False

    data_dir = resolve_pipeline_path(config.get("pipeline", {}).get("data_dir", "data"))
    dataset_name = build_dataset_name(config, start_date, end_date)
    raw_dir = data_dir / str(config.get("pipeline", {}).get("raw_dir_name", "_raw")) / dataset_name
    dataset_dir = data_dir / dataset_name

    downloaded = extract_aligned_images(
        config=config,
        start_date=start_date,
        end_date=end_date,
        raw_dir=raw_dir,
    )
    report = transform_dataset(
        config=config,
        raw_dir=raw_dir,
        dataset_dir=dataset_dir,
    )
    dvc_result = log_dataset_to_dvc(
        config=config,
        dataset_dir=dataset_dir,
        repo_root=REPO_ROOT,
    )

    print("")
    print("Dataset pipeline completed")
    print(f"Date range: {start_date.isoformat()} to {end_date.isoformat()}")
    print(f"Downloaded: {len(downloaded)}")
    print(f"Accepted: {report.accepted_count}")
    print(f"Rejected: {report.rejected_count}")
    print(f"Dataset folder: {dataset_dir}")
    if dvc_result is not None:
        print(f"DVC file: {dvc_result.dvc_file}")
        print(f"DVC pushed: {dvc_result.pushed}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a DVC dataset from MinIO aligned images.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--start-date", default="")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--no-dvc", action="store_true", help="Skip dvc add/push.")
    parser.add_argument("--no-dvc-push", action="store_true", help="Run dvc add but skip dvc push.")
    return parser.parse_args()


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a YAML object: {path}")
    return config


def parse_or_prompt_date(value: str, prompt: str) -> date:
    value = value.strip() or input(prompt).strip()
    return date.fromisoformat(value)


def build_dataset_name(config: dict[str, Any], start_date: date, end_date: date) -> str:
    template = str(
        config.get("pipeline", {}).get(
            "dataset_name_template",
            "MinIO_Dataset_{start_date}-{end_date}",
        )
    )
    return template.format(
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
    )


def resolve_pipeline_path(path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return PIPELINE_ROOT / path


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


if __name__ == "__main__":
    main()
