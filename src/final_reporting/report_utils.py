from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def relative_path(path: Path, root: Path) -> str:
    """Return a POSIX-style project-relative path when possible."""
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def require_columns(
    frame: pd.DataFrame,
    required: Iterable[str],
    source: str,
) -> None:
    """Raise when a required column is absent from a reporting input table."""
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")
