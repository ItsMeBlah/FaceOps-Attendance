from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping
from numbers import Number
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path = [
    path
    for path in sys.path
    if Path(path or os.getcwd()).resolve() != SCRIPT_DIR
]

import logging  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_EXPERIMENT_NAME = "face-recognition-training"
DEFAULT_ARTIFACT_PATH = "training_results"
DEFAULT_CONFIG_FILE = SCRIPT_DIR / "mlflow_config.yaml"
DEFAULT_MODEL_ARTIFACT_PATH = "model"
DEFAULT_REGISTERED_MODEL_NAME = "face-recognition-arcface"
HISTORY_FILE = "training_history.json"
METADATA_FILE = "training_metadata.json"


def main() -> None:
    configure_logging()
    args = parse_args()
    config = load_config(args.config)
    results_dir = resolve_results_dir(args.results_dir)

    mlflow = import_mlflow()
    configure_dagshub_mlflow(mlflow, config)
    mlflow.set_experiment(
        args.experiment_name
        or get_nested(config, "mlflow", "experiment_name")
        or DEFAULT_EXPERIMENT_NAME
    )

    metadata = load_json(results_dir / METADATA_FILE, default={})
    history = load_json(results_dir / HISTORY_FILE, default=[])
    run_name = (
        args.run_name
        or get_nested(config, "mlflow", "run_name")
        or build_run_name(results_dir, metadata)
    )
    artifact_path = (
        args.artifact_path
        or get_nested(config, "mlflow", "artifact_path")
        or DEFAULT_ARTIFACT_PATH
    )

    logger.info("Logging results folder to MLflow: %s", results_dir)
    with mlflow.start_run(run_name=run_name):
        log_metadata_params(mlflow, metadata)
        log_test_metrics_from_history(mlflow, history)
        log_result_artifacts(mlflow, results_dir, artifact_path)
        log_registered_pytorch_model(mlflow, results_dir, metadata, config)

        mlflow.set_tags(
            {
                "source": "training_module.mlflow_logging",
                "results_dir": str(results_dir),
                "model_family": "face_recognition",
            }
        )

        run = mlflow.active_run()
        if run is not None:
            logger.info("MLflow run logged: %s", run.info.run_id)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Log a face-recognition training result folder to DagsHub MLflow."
    )
    parser.add_argument(
        "results_dir",
        nargs="?",
        help="Folder containing training_history.json, training_metadata.json, and result artifacts.",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_FILE),
        help="YAML file containing DagsHub/MLflow credentials and settings.",
    )
    parser.add_argument(
        "--experiment-name",
        default=None,
        help="Optional MLflow experiment name override.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional MLflow run name override.",
    )
    parser.add_argument(
        "--artifact-path",
        default=None,
        help="Optional MLflow artifact folder override.",
    )
    return parser.parse_args()


def resolve_results_dir(results_dir: str | None) -> Path:
    if not results_dir:
        results_dir = input("Enter training results folder path: ").strip()

    path = Path(results_dir).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Results folder does not exist: {path}")

    missing = [
        file_name
        for file_name in (HISTORY_FILE, METADATA_FILE)
        if not (path / file_name).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"Results folder is missing required file(s): {', '.join(missing)}"
        )
    return path


def import_mlflow() -> Any:
    try:
        import mlflow
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "mlflow is not installed. Install it with: pip install mlflow dagshub"
        ) from exc
    return mlflow


def configure_dagshub_mlflow(mlflow: Any, config: Mapping[str, Any]) -> None:
    dagshub_config = config.get("dagshub", {})
    mlflow_config = config.get("mlflow", {})
    if not isinstance(dagshub_config, Mapping):
        dagshub_config = {}
    if not isinstance(mlflow_config, Mapping):
        mlflow_config = {}

    username = dagshub_config.get("username")
    token = dagshub_config.get("token")
    if username:
        os.environ["MLFLOW_TRACKING_USERNAME"] = str(username)
    if token:
        os.environ["MLFLOW_TRACKING_PASSWORD"] = str(token)
        os.environ["DAGSHUB_USER_TOKEN"] = str(token)

    tracking_uri = mlflow_config.get("tracking_uri")
    if tracking_uri:
        mlflow.set_tracking_uri(str(tracking_uri))
        logger.info("Using MLFLOW_TRACKING_URI=%s", tracking_uri)
        return

    repo_owner = dagshub_config.get("repo_owner")
    repo_name = dagshub_config.get("repo_name")
    repo = dagshub_config.get("repo")
    if repo and "/" in repo and (not repo_owner or not repo_name):
        repo_owner, repo_name = repo.split("/", 1)

    if not repo_owner or not repo_name:
        raise ValueError(
            "Missing DagsHub repo config. Set dagshub.repo_owner and dagshub.repo_name "
            "or mlflow.tracking_uri in mlflow_config.yaml."
        )

    try:
        import dagshub
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "dagshub is not installed. Install it with: pip install dagshub"
        ) from exc

    dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
    logger.info("Configured DagsHub MLflow repo=%s/%s", repo_owner, repo_name)


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"MLflow config file does not exist: {path}")

    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "pyyaml is not installed. Install it with: pip install pyyaml"
        ) from exc

    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    if not isinstance(config, dict):
        raise ValueError(f"MLflow config must be a YAML object: {path}")

    return config


def load_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_run_name(results_dir: Path, metadata: Mapping[str, Any]) -> str:
    output_dir = get_nested(metadata, "config", "output_dir")
    if isinstance(output_dir, str) and output_dir.strip():
        return output_dir.strip().replace("/", "_")
    return results_dir.name


def log_metadata_params(mlflow: Any, metadata: Mapping[str, Any]) -> None:
    config = metadata.get("config", {})
    dataset_metadata = metadata.get("metadata", {})

    params: dict[str, Any] = {}
    if isinstance(config, Mapping):
        params.update(flatten_params(config, prefix="config"))

    if isinstance(dataset_metadata, Mapping):
        for key in ("num_classes", "num_train", "num_val", "num_test"):
            value = dataset_metadata.get(key)
            if is_param_value(value):
                params[f"data.{key}"] = value

    if params:
        mlflow.log_params(params)
        logger.info("Logged %s metadata/config params", len(params))


def log_test_metrics_from_history(mlflow: Any, history: Any) -> None:
    if not isinstance(history, list):
        logger.warning("%s is not a list. No test metrics logged.", HISTORY_FILE)
        return

    test_row = next(
        (
            row
            for row in history
            if isinstance(row, Mapping) and str(row.get("epoch")).lower() == "test"
        ),
        None,
    )
    if test_row is None:
        logger.warning("No epoch='test' row found in %s.", HISTORY_FILE)
        return

    metrics = {
        str(key): float(value)
        for key, value in test_row.items()
        if key not in {"epoch", "lr"} and is_metric_value(value)
    }
    if metrics:
        mlflow.log_metrics(metrics)
        logger.info("Logged %s test metrics from epoch='test'", len(metrics))


def log_result_artifacts(mlflow: Any, results_dir: Path, artifact_path: str) -> None:
    artifact_files = [path for path in results_dir.rglob("*") if path.is_file()]
    model_checkpoints = {results_dir / "best.pth"}
    logged_files = [path for path in artifact_files if path not in model_checkpoints]

    for path in logged_files:
        relative_parent = path.parent.relative_to(results_dir)
        file_artifact_path = artifact_path
        if str(relative_parent) != ".":
            file_artifact_path = f"{artifact_path}/{relative_parent.as_posix()}"
        mlflow.log_artifact(str(path), artifact_path=file_artifact_path)

    mlflow.log_param("artifacts.file_count", len(logged_files))

    logger.info(
        "Logged %s artifact files from %s to artifact path '%s'",
        len(logged_files),
        results_dir,
        artifact_path,
    )


def log_registered_pytorch_model(
    mlflow: Any,
    results_dir: Path,
    metadata: Mapping[str, Any],
    config: Mapping[str, Any],
) -> None:
    model_config = config.get("model", {})
    if not isinstance(model_config, Mapping):
        model_config = {}

    checkpoint_name = str(model_config.get("checkpoint_name") or "best.pth")
    checkpoint_path = results_dir / checkpoint_name
    if not checkpoint_path.is_file():
        logger.warning("Model checkpoint was not found. Skipped model registry: %s", checkpoint_path)
        return

    _, pytorch_model = load_pytorch_model(
        checkpoint_path=checkpoint_path,
        metadata=metadata,
        normalize_output=bool(model_config.get("normalize_output", True)),
    )
    model_artifact_path = str(
        model_config.get("artifact_path") or DEFAULT_MODEL_ARTIFACT_PATH
    )
    registered_model_name = str(
        model_config.get("registered_model_name") or DEFAULT_REGISTERED_MODEL_NAME
    )

    try:
        import mlflow.pytorch as mlflow_pytorch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "MLflow PyTorch support is required. Install dependencies with: "
            "pip install mlflow torch"
        ) from exc

    mlflow_pytorch.log_model(
        pytorch_model=pytorch_model,
        artifact_path=model_artifact_path,
        registered_model_name=registered_model_name,
        serialization_format="pickle",
    )
    mlflow.log_param("model.checkpoint_name", checkpoint_name)
    mlflow.log_param(
        "model.checkpoint_size_mb",
        round(checkpoint_path.stat().st_size / 1_000_000, 3),
    )
    mlflow.set_tag("model.registered_model_name", registered_model_name)
    logger.info(
        "Logged PyTorch model to registry name=%s artifact_path=%s checkpoint=%s",
        registered_model_name,
        model_artifact_path,
        checkpoint_path,
    )


def load_pytorch_model(
    checkpoint_path: Path,
    metadata: Mapping[str, Any],
    normalize_output: bool,
) -> tuple[Any, Any]:
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "torch is required to log the PyTorch model. Install it before running this logger."
        ) from exc

    face_recognition_root = SCRIPT_DIR.parent / "face_recognition_module"
    if str(face_recognition_root) not in sys.path:
        sys.path.insert(0, str(face_recognition_root))

    from resnet18_training.model import resnet18_face

    class EmbeddingModelWrapper(nn.Module):
        def __init__(self, backbone: nn.Module, normalize: bool) -> None:
            super().__init__()
            self.backbone = backbone
            self.normalize = normalize

        def forward(self, image):
            embedding = self.backbone(image)
            if self.normalize:
                embedding = F.normalize(embedding, p=2, dim=1)
            return embedding

    model_config = metadata.get("config", {})
    if not isinstance(model_config, Mapping):
        model_config = {}

    input_size = int(model_config.get("image_size", 128))
    embedding_size = int(model_config.get("embedding_size", 512))
    use_se = bool(model_config.get("use_se", False))
    dropout = float(model_config.get("dropout", 0.0))

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state", checkpoint)
    if any(key.startswith("module.") for key in state_dict):
        state_dict = {key.replace("module.", "", 1): value for key, value in state_dict.items()}

    backbone = resnet18_face(
        input_size=input_size,
        embedding_size=embedding_size,
        input_channels=1,
        use_se=use_se,
        dropout=dropout,
    )
    backbone.load_state_dict(state_dict)
    model = EmbeddingModelWrapper(backbone, normalize=normalize_output)
    model.eval()
    return torch, model


def flatten_params(
    values: Mapping[str, Any],
    prefix: str,
    max_value_length: int = 250,
) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for key, value in values.items():
        param_key = f"{prefix}.{key}"
        if isinstance(value, Mapping):
            params.update(flatten_params(value, param_key, max_value_length=max_value_length))
        elif is_param_value(value):
            params[param_key] = trim_param_value(value, max_value_length)
    return params


def trim_param_value(value: Any, max_value_length: int) -> Any:
    if isinstance(value, str) and len(value) > max_value_length:
        return value[: max_value_length - 3] + "..."
    return value


def is_param_value(value: Any) -> bool:
    return isinstance(value, (str, bool, int, float)) or value is None


def is_metric_value(value: Any) -> bool:
    return isinstance(value, Number) and not isinstance(value, bool)


def get_nested(values: Mapping[str, Any], *keys: str) -> Any:
    current: Any = values
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def configure_logging() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


if __name__ == "__main__":
    main()
