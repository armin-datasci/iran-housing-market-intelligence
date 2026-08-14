from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import polars as pl

from src.milestone_4.gold.contracts import (
    EXPECTED_DIMENSIONS,
    EXPECTED_MARTS,
    FORBIDDEN_COLUMN_TOKENS,
    REQUIRED_ARTIFACT_COLUMNS,
)


DEFAULT_GOLD = PROJECT_ROOT / "data" / "gold"
DEFAULT_DEST = (PROJECT_ROOT/ "src"/ "api"/ "deployment"/ "api_data"/ "gold")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stage the compact canonical Gold artifacts required by the IHMI "
            "FastAPI deployment."
        )
    )
    parser.add_argument("--gold-dir", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--reset", action="store_true")
    return parser.parse_args()


def validate_parquet(path: Path, artifact_name: str) -> tuple[int, list[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing canonical Gold artifact: {path}")

    scan = pl.scan_parquet(path)
    columns = scan.collect_schema().names()
    required = set(REQUIRED_ARTIFACT_COLUMNS[artifact_name])
    missing = sorted(required - set(columns))
    if missing:
        raise ValueError(f"{artifact_name} is missing required columns: {missing}")

    forbidden = sorted(
        col
        for col in columns
        if any(token in col.lower() for token in FORBIDDEN_COLUMN_TOKENS)
    )
    if forbidden:
        raise ValueError(
            f"{artifact_name} contains API-forbidden spatial/legacy columns: {forbidden}"
        )

    rows = int(scan.select(pl.len().alias("n")).collect().item())
    return rows, columns


def main() -> int:
    args = parse_args()
    gold_dir = args.gold_dir.resolve()
    destination = args.destination.resolve()

    if args.reset and destination.exists():
        shutil.rmtree(destination)

    marts_dest = destination / "marts"
    dims_dest = destination / "dimensions"
    qa_dest = destination / "qa"
    for directory in (marts_dest, dims_dest, qa_dest):
        directory.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, object]] = []

    for name in EXPECTED_MARTS:
        src = gold_dir / "marts" / f"{name}.parquet"
        rows, columns = validate_parquet(src, name)
        dst = marts_dest / src.name
        shutil.copy2(src, dst)
        manifest_rows.append(
            {
                "artifact": name,
                "kind": "mart",
                "rows": rows,
                "columns": len(columns),
                "staged_path": (
                    dst.relative_to(PROJECT_ROOT).as_posix()
                    if dst.is_relative_to(PROJECT_ROOT)
                    else dst.as_posix()
                ),
            }
        )

    for name in EXPECTED_DIMENSIONS:
        src = gold_dir / "dimensions" / f"{name}.parquet"
        rows, columns = validate_parquet(src, name)
        dst = dims_dest / src.name
        shutil.copy2(src, dst)
        manifest_rows.append(
            {
                "artifact": name,
                "kind": "dimension",
                "rows": rows,
                "columns": len(columns),
                "staged_path": (
                    dst.relative_to(PROJECT_ROOT).as_posix()
                    if dst.is_relative_to(PROJECT_ROOT)
                    else dst.as_posix()
                ),
            }
        )

    for qa_name in ("gold_qa_manifest.json", "gold_manifest.json"):
        src = gold_dir / "qa" / qa_name
        if src.exists():
            shutil.copy2(src, qa_dest / qa_name)

    manifest = {
        "deployment_data_contract": "ihmi-fastapi-gold-readonly-v1",
        "gold_source": "canonical data/gold",
        "marts": len(EXPECTED_MARTS),
        "dimensions": len(EXPECTED_DIMENSIONS),
        "artifacts": manifest_rows,
        "privacy": (
            "No exact coordinate columns are permitted in the staged deployment bundle."
        ),
    }
    (destination / "api_bundle_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("IHMI FASTAPI DEPLOYMENT DATA STAGED")
    print("destination:", destination)
    print("marts:", len(EXPECTED_MARTS))
    print("dimensions:", len(EXPECTED_DIMENSIONS))
    print("artifacts:", len(manifest_rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
