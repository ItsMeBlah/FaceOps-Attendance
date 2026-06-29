from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DVCResult:
    dataset_path: str
    dvc_file: str
    pushed: bool


def log_dataset_to_dvc(
    config: dict[str, Any],
    dataset_dir: Path,
    repo_root: Path,
) -> DVCResult | None:
    dvc_config = config.get("dvc", {})
    if not bool(dvc_config.get("enabled", True)):
        logger.info("DVC logging is disabled.")
        return None

    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset folder does not exist: {dataset_dir}")

    dataset_path = dataset_dir.resolve()
    relative_dataset_path = dataset_path.relative_to(repo_root.resolve())
    run_command(["dvc", "add", str(relative_dataset_path)], cwd=repo_root)

    pushed = False
    if bool(dvc_config.get("push", True)):
        remote = str(dvc_config.get("remote", "origin"))
        run_command(["dvc", "push", "-r", remote, str(relative_dataset_path)], cwd=repo_root)
        pushed = True

    dvc_file = repo_root / f"{relative_dataset_path}.dvc"
    logger.info("DVC dataset ready path=%s dvc_file=%s pushed=%s", dataset_dir, dvc_file, pushed)
    return DVCResult(
        dataset_path=str(dataset_dir),
        dvc_file=str(dvc_file),
        pushed=pushed,
    )


def run_command(command: list[str], cwd: Path) -> None:
    logger.info("Running command: %s", " ".join(command))
    try:
        subprocess.run(command, cwd=cwd, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError("dvc is not installed or is not available on PATH.") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Command failed: {' '.join(command)}") from exc
