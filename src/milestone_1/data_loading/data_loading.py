from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import polars as pl

from src.common.config import configured_path
from src.common.paths import relative_to_project
from src.common.validation import Check, make_check, summarize_checks

try:
    import psutil
except ImportError:
    psutil = None


def _resource_snapshot() -> dict[str, float | int | None]:
    if psutil is None:
        return {"ram_mib": None, "thread_count": None}
    process = psutil.Process()
    return {
        "ram_mib": process.memory_info().rss / (1024**2),
        "thread_count": process.num_threads(),
    }


def inspect_raw_data(
    raw_path: Path | None = None,
    *,
    expected_columns: int | None = None,
    expected_rows: int | None = None,
) -> dict[str, Any]:
    """Inspect the complete Raw CSV without cleaning, sampling, or persistence."""
    path = (raw_path or configured_path("raw_source")).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Raw source not found: {path}")

    started = perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    before = _resource_snapshot()

    scan = pl.scan_csv(
        path,
        has_header=True,
        infer_schema=False,
        schema_overrides=None,
        try_parse_dates=False,
        ignore_errors=False,
        truncate_ragged_lines=False,
        low_memory=True,
    )
    schema = scan.collect_schema()
    columns = schema.names()
    row_count = int(scan.select(pl.len()).collect(engine="streaming").item())

    checks: list[Check] = []
    if expected_columns is not None:
        checks.append(make_check("raw_column_count", "schema", len(columns), expected_columns, len(columns) == expected_columns))
    if expected_rows is not None:
        checks.append(make_check("raw_row_count", "schema", row_count, expected_rows, row_count == expected_rows, review_on_fail=True, critical=False, notes="Row count is a source snapshot diagnostic, not a fixed business contract."))
    checks.append(make_check("raw_schema_nonempty", "schema", len(columns), ">0", len(columns) > 0))

    summary = summarize_checks(checks)
    return {
        "path": path,
        "row_count": row_count,
        "column_count": len(columns),
        "columns": columns,
        "schema": {name: str(dtype) for name, dtype in schema.items()},
        "checks": checks,
        "status": summary,
        "manifest": {
            "stage": "m1_raw_inspection",
            "created_at_utc": started_at,
            "source_path": relative_to_project(path),
            "size_bytes": path.stat().st_size,
            "row_count": row_count,
            "column_count": len(columns),
            "elapsed_seconds": round(perf_counter() - started, 4),
            "initial_ram_mib": before["ram_mib"],
            "final_ram_mib": _resource_snapshot()["ram_mib"],
        },
    }

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect the external Raw source without modifying it.")
    parser.add_argument("--raw", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = inspect_raw_data(args.raw)
    print("M1 RAW INSPECTION COMPLETED")
    print(f"path: {relative_to_project(result['path'])}")
    print(f"rows: {result['row_count']:,}")
    print(f"columns: {result['column_count']}")
    print(f"status: {result['status']['overall_status']}")


if __name__ == "__main__":
    main()
