from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

import polars as pl


def _atomic_target(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.with_name(f".{path.name}.{os.getpid()}.tmp")


def atomic_write_text(text: str, path: Path) -> None:
    tmp = _atomic_target(path)
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_json(payload: Mapping[str, Any], path: Path) -> None:
    atomic_write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n", path)


def atomic_write_csv(frame: pl.DataFrame, path: Path) -> None:
    tmp = _atomic_target(path)
    frame.write_csv(tmp)
    os.replace(tmp, path)


def atomic_sink_parquet(frame: pl.LazyFrame, path: Path, *, compression: str = "zstd") -> None:
    tmp = _atomic_target(path)
    frame.sink_parquet(tmp, compression=compression, statistics=True, maintain_order=True, engine="streaming")
    os.replace(tmp, path)
