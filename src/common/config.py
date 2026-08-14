from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from .paths import CONFIG_DIR, resolve_project_path


@lru_cache(maxsize=1)
def load_settings(path: Path = CONFIG_DIR / "settings.yaml") -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError("settings.yaml must contain a mapping at the root.")
    return data


def setting(*keys: str, default: Any = None) -> Any:
    value: Any = load_settings()
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def configured_path(key: str) -> Path:
    value = setting("paths", key)
    if not isinstance(value, str) or not value.strip():
        raise KeyError(f"Missing paths.{key} in config/settings.yaml")
    return resolve_project_path(value)


def price_unit() -> str:
    return str(setting("contracts", "price_unit", default="toman_assumed_unconfirmed"))


def price_observation_type() -> str:
    return str(setting("contracts", "price_observation_type", default="asking_price"))
