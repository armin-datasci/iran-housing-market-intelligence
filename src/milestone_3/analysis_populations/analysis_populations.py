from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import polars as pl

from src.common.config import configured_path
from src.common.io_utils import atomic_sink_parquet, atomic_write_csv, atomic_write_json
from src.common.paths import OUTPUTS_DIR, relative_to_project
from src.common.validation import Check, checks_frame, make_check, summarize_checks

VERSION = "m3-analysis-populations-v1"
PROGRESS_WIDTH = 30

BASE_COLUMNS: dict[str, list[str]] = {
    "sales": [
        "source_row_id", "analysis_month", "city_slug", "neighborhood_slug", "cat3_slug",
        "property_family", "user_type", "price_regime", "primary_area_sqm",
        "sale_price_final_toman", "sale_price_per_sqm_final_toman", "rooms_count_num",
        "building_age_years", "construction_year_before_1370_flag", "floor_num",
        "total_floors_count_num", "has_elevator_bool", "has_parking_bool",
        "has_warehouse_bool", "has_balcony_bool", "is_rebuilt_bool", "building_direction",
        "title_normalized", "description_normalized", "latitude", "longitude",
        "quality_issue_count", "record_quality_review_flag", "price_unit",
        "price_observation_type", "silver_master_version", "quality_rule_version",
        "sale_price_per_sqm_final_eligible_flag", "supply_keep_flag",
    ],
    "rent": [
        "source_row_id", "analysis_month", "city_slug", "neighborhood_slug", "cat3_slug",
        "property_family", "user_type", "price_regime", "primary_area_sqm",
        "monthly_rent_final_toman", "deposit_final_toman",
        "rent_equivalent_monthly_low_toman", "rent_equivalent_monthly_base_toman",
        "rent_equivalent_monthly_high_toman", "rooms_count_num", "building_age_years",
        "floor_num", "has_elevator_bool", "has_parking_bool", "has_warehouse_bool",
        "has_balcony_bool", "quality_issue_count", "record_quality_review_flag",
        "price_unit", "price_observation_type", "silver_master_version",
        "rent_final_eligible_flag", "supply_keep_flag",
    ],
    "supply": [
        "source_row_id", "analysis_month", "city_slug", "neighborhood_slug", "cat3_slug",
        "property_family", "user_type", "price_regime", "supply_keep_flag",
        "exact_duplicate_flag", "same_month_duplicate_flag", "probable_duplicate_cluster_id",
        "deduplication_confidence", "quality_issue_count", "record_quality_review_flag",
        "silver_master_version", "supply_analysis_eligible_flag",
    ],
    "map": [
        "source_row_id", "analysis_month", "city_slug", "neighborhood_slug", "cat3_slug",
        "property_family", "price_regime", "latitude", "longitude", "location_radius_m",
        "coordinate_pair_present", "coordinate_partial_flag", "geo_country_valid",
        "quality_issue_count", "record_quality_review_flag", "silver_master_version",
        "map_analysis_eligible_flag", "geo_aggregate_map_eligible_flag", "supply_keep_flag",
    ],
}

REQUIRED = {
    # Identity / grouping fields used across M3.
    "source_row_id", "analysis_month", "city_slug", "neighborhood_slug", "cat3_slug",
    "property_family", "user_type", "price_regime", "is_core_analysis_period",
    # Canonical M2 metrics and eligibility. M3 must consume, never recreate, these fields.
    "primary_area_sqm", "sale_price_per_sqm_final_toman",
    "sale_price_per_sqm_final_eligible_flag", "rent_final_eligible_flag",
    "supply_analysis_eligible_flag", "geo_aggregate_map_eligible_flag", "supply_keep_flag",
    # Fields required by drivers / seller / text / segmentation.
    "rooms_count_num", "building_age_years", "construction_year_before_1370_flag",
    "floor_num", "has_elevator_bool", "has_parking_bool", "has_warehouse_bool",
    "has_balcony_bool", "building_direction", "title_normalized", "description_normalized",
    # Spatial and interpretation contract.
    "latitude", "longitude", "coordinate_pair_present", "coordinate_partial_flag",
    "geo_country_valid", "price_unit", "price_observation_type",
    "silver_master_version", "quality_rule_version",
}


def show_progress(percent: int, label: str, *, final: bool = False) -> None:
    pct = max(0, min(100, int(percent)))
    filled = int(PROGRESS_WIDTH * pct / 100)
    bar = "#" * filled + "-" * (PROGRESS_WIDTH - filled)
    print(
        f"\rM3 populations [{bar}] {pct:3d}% complete | {100-pct:3d}% remaining | {label[:42]:42s}",
        end="\n" if final else "",
        flush=True,
    )


def _bool(name: str) -> pl.Expr:
    return pl.col(name).cast(pl.Boolean, strict=False).fill_null(False)


def _base_filter(name: str, columns: set[str]) -> pl.Expr:
    if name == "sales":
        return _bool("sale_price_per_sqm_final_eligible_flag")
    if name == "rent":
        return _bool("rent_final_eligible_flag")
    if name == "supply":
        return _bool("supply_analysis_eligible_flag")
    if "map_analysis_eligible_flag" in columns:
        return _bool("map_analysis_eligible_flag")
    return _bool("geo_aggregate_map_eligible_flag") & _bool("supply_keep_flag")


def _selected(name: str, columns: set[str]) -> list[str]:
    return [column for column in BASE_COLUMNS[name] if column in columns]


def _stats(path: Path) -> dict[str, Any]:
    scan = pl.scan_parquet(path)
    schema = scan.collect_schema()
    month_exprs: list[pl.Expr] = [pl.len().alias("rows")]
    if "analysis_month" in schema.names():
        month_exprs += [pl.col("analysis_month").min().alias("min_month"), pl.col("analysis_month").max().alias("max_month")]
    row = scan.select(month_exprs).collect(engine="streaming").row(0, named=True)
    return {
        "row_count": int(row["rows"]),
        "column_count": len(schema),
        "min_month": str(row.get("min_month")) if row.get("min_month") is not None else None,
        "max_month": str(row.get("max_month")) if row.get("max_month") is not None else None,
    }


def run(silver_path: Path | None = None) -> dict[str, Path]:
    silver = (silver_path or configured_path("silver_master")).resolve()
    if not silver.exists():
        raise FileNotFoundError(f"Silver Master not found: {silver}")

    output_dir = OUTPUTS_DIR / "model_artifacts" / "milestone_3" / "analysis_populations"
    table_dir = OUTPUTS_DIR / "tables" / "milestone_3" / "analysis_populations"
    qa_dir = OUTPUTS_DIR / "qa" / "milestone_3" / "analysis_populations"
    output_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    qa_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    show_progress(0, "reading Silver schema")
    scan = pl.scan_parquet(silver)
    columns = set(scan.collect_schema().names())
    missing = sorted(REQUIRED - columns)
    if missing:
        raise ValueError(f"Silver Master is missing required M3 population columns: {missing}")

    checks: list[Check] = [
        make_check("silver_required_columns", "input", len(missing), 0, not missing),
    ]
    outputs: dict[str, Path] = {}
    summary_rows: list[dict[str, Any]] = []

    for idx, name in enumerate(["sales", "rent", "supply", "map"], start=1):
        selected = _selected(name, columns)
        target = output_dir / f"{name}_analysis_base.parquet"
        base = scan.filter(_base_filter(name, columns)).select(selected)
        atomic_sink_parquet(base, target)
        stats = _stats(target)
        summary_rows.append({"population": name, **stats, "source": "silver_master", "version": VERSION})
        checks.append(
            make_check(
                f"{name}_population_nonempty", "population", stats["row_count"], ">0",
                stats["row_count"] > 0,
            )
        )
        outputs[name] = target
        show_progress(10 + idx * 18, f"{name} base: {stats['row_count']:,} rows")

    summary = pl.DataFrame(summary_rows)
    summary_path = table_dir / "analysis_population_summary.csv"
    checks_path = qa_dir / "analysis_population_checks.csv"
    manifest_path = qa_dir / "analysis_population_manifest.json"
    atomic_write_csv(summary, summary_path)
    atomic_write_csv(checks_frame(checks), checks_path)
    status = summarize_checks(checks)
    atomic_write_json(
        {
            "version": VERSION,
            "status": status,
            "input": relative_to_project(silver),
            "outputs": {key: relative_to_project(value) for key, value in outputs.items()},
            "summary": relative_to_project(summary_path),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        },
        manifest_path,
    )
    outputs.update({"summary": summary_path, "checks": checks_path, "manifest": manifest_path})
    show_progress(100, f"complete in {time.perf_counter()-started:.1f}s", final=True)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build canonical M3 analysis populations from Silver Master.")
    parser.add_argument("--silver", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run(args.silver)
    print("M3 ANALYSIS POPULATIONS COMPLETED")
    for name, path in outputs.items():
        print(f"{name}: {relative_to_project(path)}")


if __name__ == "__main__":
    main()
