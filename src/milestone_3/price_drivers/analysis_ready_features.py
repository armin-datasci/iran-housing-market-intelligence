from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import polars as pl

from src.common.io_utils import atomic_sink_parquet, atomic_write_csv, atomic_write_json
from src.common.paths import OUTPUTS_DIR, relative_to_project
from src.common.validation import Check, checks_frame, make_check, summarize_checks

VERSION = "m3-analysis-ready-v1"
PROGRESS_WIDTH = 30
SALES_BASE = OUTPUTS_DIR / "model_artifacts" / "milestone_3" / "analysis_populations" / "sales_analysis_base.parquet"
OUTPUT_PARQUET = OUTPUTS_DIR / "model_artifacts" / "milestone_3" / "price_drivers" / "analysis_ready_features.parquet"
TARGET = "sale_price_per_sqm_final_toman"

NUMERIC_FEATURES = ["primary_area_sqm", "rooms_count_num", "building_age_years", "floor_num", "total_floors_count_num"]
BINARY_FEATURES = [
    "construction_year_before_1370_flag", "has_elevator_bool", "has_parking_bool",
    "has_warehouse_bool", "has_balcony_bool", "is_rebuilt_bool",
]
CATEGORICAL_FEATURES = ["analysis_month", "city_slug", "neighborhood_slug", "cat3_slug", "property_family", "building_direction"]
IDENTITY_COLUMNS = ["source_row_id", "user_type", "price_unit", "price_observation_type", "silver_master_version", "quality_rule_version"]


def show_progress(percent: int, label: str, *, final: bool = False) -> None:
    pct = max(0, min(100, int(percent)))
    filled = int(PROGRESS_WIDTH * pct / 100)
    bar = "#" * filled + "-" * (PROGRESS_WIDTH - filled)
    print(
        f"\rM3 features [{bar}] {pct:3d}% complete | {100-pct:3d}% remaining | {label[:42]:42s}",
        end="\n" if final else "", flush=True,
    )


def _split_expr() -> pl.Expr:
    month = pl.col("analysis_month").cast(pl.String).str.slice(0, 7)
    return (
        pl.when(month.is_in(["2024-05", "2024-06", "2024-07", "2024-08", "2024-09", "2024-10"]))
        .then(pl.lit("train"))
        .when(month == "2024-11").then(pl.lit("validation"))
        .when(month == "2024-12").then(pl.lit("test"))
        .otherwise(pl.lit("outside_core_split"))
        .alias("analysis_split")
    )


def run(sales_base: Path = SALES_BASE) -> dict[str, Path]:
    sales_base = sales_base.resolve()
    if not sales_base.exists():
        raise FileNotFoundError(f"Sales analysis base not found: {sales_base}")
    output_path = OUTPUT_PARQUET
    table_dir = OUTPUTS_DIR / "tables" / "milestone_3" / "price_drivers"
    qa_dir = OUTPUTS_DIR / "qa" / "milestone_3" / "price_drivers"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    qa_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    show_progress(0, "validating sales population")
    scan = pl.scan_parquet(sales_base)
    columns = set(scan.collect_schema().names())
    required = {"source_row_id", "analysis_month", TARGET, "city_slug", "cat3_slug", "property_family", "primary_area_sqm"}
    missing = sorted(required - columns)
    if missing:
        raise ValueError(f"Sales base missing analysis-ready columns: {missing}")

    selected = [column for column in [*IDENTITY_COLUMNS, TARGET, *NUMERIC_FEATURES, *BINARY_FEATURES, *CATEGORICAL_FEATURES] if column in columns]
    target = pl.col(TARGET).cast(pl.Float64, strict=False)
    work = (
        scan.select(selected)
        .filter(target.is_not_null() & target.is_finite() & (target > 0))
        .with_columns([
            target.log().alias("log_sale_price_per_sqm"),
            pl.col("primary_area_sqm").cast(pl.Float64, strict=False).log().alias("log_primary_area_sqm"),
            _split_expr(),
            pl.lit(VERSION).alias("analysis_ready_version"),
        ])
    )
    atomic_sink_parquet(work, output_path)
    show_progress(55, "analysis-ready parquet written")

    out_scan = pl.scan_parquet(output_path)
    out_schema = out_scan.collect_schema().names()
    stats = out_scan.select([
        pl.len().alias("row_count"),
        pl.col("source_row_id").n_unique().alias("unique_source_rows"),
        pl.col("analysis_split").filter(pl.col("analysis_split") == "train").len().alias("train_rows"),
        pl.col("analysis_split").filter(pl.col("analysis_split") == "validation").len().alias("validation_rows"),
        pl.col("analysis_split").filter(pl.col("analysis_split") == "test").len().alias("test_rows"),
    ]).collect(engine="streaming").row(0, named=True)

    missing_exprs: list[pl.Expr] = []
    feature_cols = [column for column in [*NUMERIC_FEATURES, *BINARY_FEATURES, *CATEGORICAL_FEATURES] if column in out_schema]
    for column in feature_cols:
        missing_exprs.append(pl.col(column).is_null().sum().alias(column))
    missing_values = out_scan.select(missing_exprs).collect(engine="streaming").row(0, named=True) if missing_exprs else {}
    n = int(stats["row_count"] or 0)
    missingness = pl.DataFrame([
        {"feature": column, "missing_count": int(missing_values.get(column, 0) or 0), "missing_rate": (int(missing_values.get(column, 0) or 0) / n if n else None)}
        for column in feature_cols
    ])

    role_rows: list[dict[str, Any]] = []
    for column in out_schema:
        if column == TARGET or column == "log_sale_price_per_sqm":
            role, model = "target", False
        elif column in NUMERIC_FEATURES or column == "log_primary_area_sqm":
            role, model = "numeric_feature", True
        elif column in BINARY_FEATURES:
            role, model = "binary_feature", True
        elif column in CATEGORICAL_FEATURES:
            role, model = ("control" if column in {"analysis_month", "city_slug", "neighborhood_slug"} else "categorical_feature"), True
        else:
            role, model = "metadata", False
        role_rows.append({"column": column, "role": role, "allowed_in_primary_model": model})
    contract = pl.DataFrame(role_rows)

    checks: list[Check] = [
        make_check("analysis_ready_nonempty", "features", n, ">0", n > 0),
        make_check("source_row_id_unique", "features", stats["unique_source_rows"], n, int(stats["unique_source_rows"] or 0) == n),
        make_check("train_split_nonempty", "features", stats["train_rows"], ">0", int(stats["train_rows"] or 0) > 0),
        make_check("validation_split_nonempty", "features", stats["validation_rows"], ">0", int(stats["validation_rows"] or 0) > 0),
        make_check("test_split_nonempty", "features", stats["test_rows"], ">0", int(stats["test_rows"] or 0) > 0),
        make_check(
            "text_excluded_from_primary_model", "leakage", "title/description not selected", "excluded", True,
            notes="NLP signals are evaluated separately and are not primary clustering/price-driver inputs.",
        ),
    ]

    contract_path = table_dir / "analysis_ready_feature_contract.csv"
    missingness_path = qa_dir / "analysis_ready_missingness.csv"
    checks_path = qa_dir / "analysis_ready_checks.csv"
    manifest_path = qa_dir / "analysis_ready_manifest.json"
    atomic_write_csv(contract, contract_path)
    atomic_write_csv(missingness, missingness_path)
    atomic_write_csv(checks_frame(checks), checks_path)
    status = summarize_checks(checks)
    atomic_write_json(
        {
            "version": VERSION,
            "status": status,
            "input": relative_to_project(sales_base),
            "output": relative_to_project(output_path),
            "row_count": n,
            "column_count": len(out_schema),
            "split_counts": {"train": int(stats["train_rows"] or 0), "validation": int(stats["validation_rows"] or 0), "test": int(stats["test_rows"] or 0)},
            "imputation_contract": "Numeric/binary imputation is fit inside each model training fold/split; missing is not blanket-filled in this artifact.",
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }, manifest_path,
    )
    show_progress(100, f"complete: {n:,} rows / {len(out_schema)} columns", final=True)
    return {"features": output_path, "contract": contract_path, "missingness": missingness_path, "checks": checks_path, "manifest": manifest_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build M3 analysis-ready sale features from canonical sales population.")
    parser.add_argument("--sales-base", type=Path, default=SALES_BASE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run(args.sales_base)
    print("M3 ANALYSIS-READY FEATURES COMPLETED")
    for name, path in outputs.items():
        print(f"{name}: {relative_to_project(path)}")


if __name__ == "__main__":
    main()
