from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


BACKEND_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = BACKEND_ROOT / "app/configs/config.yaml"
DEFAULT_WEIGHTS_DIR = BACKEND_ROOT / "weights"
DEFAULT_CAPTURES_DIR = BACKEND_ROOT / "captures" / "screenshots"


def _bool_from_env(name: str, default: bool = False) -> bool:
	value = os.getenv(name)
	if value is None:
		return default
	return _to_bool(value)


def _to_bool(value: Any) -> bool:
	if isinstance(value, bool):
		return value
	if isinstance(value, str):
		return value.strip().lower() in {"1", "true", "yes", "y", "on"}
	return bool(value)


def _load_yaml_config() -> dict[str, Any]:
	config_path = Path(os.getenv("BACKEND_CONFIG_PATH", DEFAULT_CONFIG_PATH)).expanduser()
	if not config_path.is_file():
		return {}

	with config_path.open("r", encoding="utf-8") as handle:
		config = yaml.safe_load(handle) or {}

	if not isinstance(config, dict):
		raise ValueError(f"Config file must contain a YAML object: {config_path}")
	return config


def _get(config: dict[str, Any], *keys: str, default: Any = None) -> Any:
	value: Any = config
	for key in keys:
		if not isinstance(value, dict) or key not in value:
			return default
		value = value[key]
	return value


def _resolve_backend_path(path_value: str | Path) -> Path:
	path = Path(path_value).expanduser()
	if path.is_absolute():
		return path
	return BACKEND_ROOT / path


_config = _load_yaml_config()


@dataclass(frozen=True)
class Settings:
	use_triton: bool = _bool_from_env(
		"BACKEND_USE_TRITON",
		_to_bool(_get(_config, "inference", "use_triton", default=False)),
	)
	triton_url: str = os.getenv(
		"TRITON_URL",
		str(_get(_config, "triton", "url", default="localhost:8000")),
	)
	model_version: str = os.getenv(
		"TRITON_MODEL_VERSION",
		str(_get(_config, "triton", "model_version", default="")),
	)
	request_timeout: int = int(
		float(
			os.getenv(
				"TRITON_REQUEST_TIMEOUT",
				str(_get(_config, "triton", "request_timeout", default=10)),
			)
		)
	)
	weights_dir: Path = _resolve_backend_path(
		os.getenv(
			"BACKEND_WEIGHTS_DIR",
			_get(_config, "models", "weights_dir", default=DEFAULT_WEIGHTS_DIR),
		)
	)
	captures_enabled: bool = _bool_from_env(
		"BACKEND_SAVE_CAPTURES",
		_to_bool(_get(_config, "captures", "enabled", default=False)),
	)
	captures_dir: Path = _resolve_backend_path(
		os.getenv(
			"BACKEND_CAPTURES_DIR",
			_get(_config, "captures", "dir", default=DEFAULT_CAPTURES_DIR),
		)
	)
	qdrant_url: str = os.getenv(
		"QDRANT_URL",
		str(_get(_config, "qdrant", "url", default="http://localhost:6333")),
	)
	qdrant_api_key: str = os.getenv(
		"QDRANT_API_KEY",
		str(_get(_config, "qdrant", "api_key", default="")),
	)
	qdrant_collection: str = os.getenv(
		"QDRANT_COLLECTION",
		str(_get(_config, "qdrant", "collection", default="face_embeddings")),
	)
	qdrant_vector_size: int = int(
		os.getenv(
			"QDRANT_VECTOR_SIZE",
			str(_get(_config, "qdrant", "vector_size", default=512)),
		)
	)
	qdrant_top_k: int = int(
		os.getenv(
			"QDRANT_TOP_K",
			str(_get(_config, "qdrant", "top_k", default=5)),
		)
	)
	qdrant_max_registration_images: int = int(
		os.getenv(
			"QDRANT_MAX_REGISTRATION_IMAGES",
			str(_get(_config, "qdrant", "max_registration_images", default=5)),
		)
	)
	mongodb_uri: str = os.getenv(
		"MONGODB_URI",
		str(
			_get(
				_config,
				"mongodb",
				"uri",
				default="mongodb://localhost:27017",
			)
		),
	)
	mongodb_database: str = os.getenv(
		"MONGODB_DATABASE",
		str(_get(_config, "mongodb", "database", default="faceguard")),
	)
	mongodb_timeout_ms: int = int(
		os.getenv(
			"MONGODB_TIMEOUT_MS",
			str(_get(_config, "mongodb", "timeout_ms", default=5000)),
		)
	)


settings = Settings()
