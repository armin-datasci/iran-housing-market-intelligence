from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _project_root() -> Path:
    override = os.getenv("IHMI_PROJECT_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


def _gold_dir(project_root: Path) -> Path:
    override = os.getenv("IHMI_GOLD_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return project_root / "data" / "gold"


def _cors_origins() -> tuple[str, ...]:
    raw = os.getenv("IHMI_API_CORS_ORIGINS", "")
    return tuple(x.strip() for x in raw.split(",") if x.strip())


@dataclass(frozen=True, slots=True)
class ApiSettings:
    project_root: Path
    gold_dir: Path
    api_prefix: str
    cors_origins: tuple[str, ...]
    default_page_size: int
    max_page_size: int


def load_settings() -> ApiSettings:
    project_root = _project_root()
    return ApiSettings(
        project_root=project_root,
        gold_dir=_gold_dir(project_root),
        api_prefix=os.getenv("IHMI_API_PREFIX", "/api/v1").rstrip("/"),
        cors_origins=_cors_origins(),
        default_page_size=max(1, int(os.getenv("IHMI_API_DEFAULT_PAGE_SIZE", "100"))),
        max_page_size=max(1, int(os.getenv("IHMI_API_MAX_PAGE_SIZE", "500"))),
    )
