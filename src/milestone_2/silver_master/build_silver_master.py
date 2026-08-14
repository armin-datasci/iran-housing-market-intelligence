from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from time import perf_counter
from typing import Any

import polars as pl
import yaml

from src.common.config import configured_path, price_observation_type, price_unit
from src.common.io_utils import atomic_sink_parquet, atomic_write_csv, atomic_write_text
from src.common.paths import CONFIG_DIR, DOCS_DIR, OUTPUTS_DIR, relative_to_project
from src.milestone_2.cleaning_management.duplicates.duplicate_validation import add_duplicate_features, write_duplicate_reports
from src.milestone_2.cleaning_management.missingness.missingness_audit import run_missingness_audit
from src.milestone_2.cleaning_management.outliers.outlier_rules import apply_outlier_rules
from src.milestone_2.cleaning_management.outliers.outlier_validation import run_outlier_validation
from src.milestone_2.cleaning_management.standardization.structured_standardization import (
    BOOLEAN_COLUMNS,
    MONEY_COLUMNS,
    apply_structured_standardization,
    write_standardization_summary,
)
from src.milestone_2.cleaning_management.standardization.text_standardization import add_normalized_text_columns
from src.milestone_2.currency.currency_validation import run_currency_validation
from src.milestone_2.final_metrics.final_metric_validation import run_final_metric_validation
from src.milestone_2.final_metrics.price_metrics import apply_final_metrics
from src.milestone_2.price_regimes.price_regimes import apply_price_regimes, write_price_regime_review
from src.milestone_2.quality_gate.quality_gate import run_quality_gate

PROGRESS_WIDTH = 30

DERIVED_COLUMNS = [
    "source_row_id", "analysis_month", "title_normalized", "description_normalized", "property_family",
    "building_size_sqm", "land_size_sqm", "latitude", "longitude", "location_radius_m",
    "regular_person_capacity_num", "extra_person_capacity_num", "rooms_count_num",
    "construction_year_jalali", "building_age_years", "floor_num", "total_floors_count_num", "unit_per_floor_num",
    "rooms_count_censored_flag", "construction_year_before_1370_flag", "construction_year_numeric_invalid_flag",
    "floor_censored_flag", "total_floors_count_censored_flag", "total_floors_count_unselected_flag",
    "unit_per_floor_censored_flag", "unit_per_floor_unselected_flag", "has_balcony_unselected_flag",
    *MONEY_COLUMNS.values(), *[f"{column}_bool" for column in BOOLEAN_COLUMNS],
    "type_parse_error_count", "is_core_analysis_period", "is_sale", "is_long_term_rent", "is_temporary_rent",
    "market_regime", "coordinate_pair_present", "coordinate_partial_flag", "geo_country_valid",
    "area_basis", "primary_area_sqm", "sale_price_status", "sale_price_clean_toman",
    "rent_component_status", "credit_component_status", "long_term_rent_regime", "price_per_sqm_raw_toman",
    "monthly_rent_clean_toman", "deposit_clean_toman", "price_regime", "price_regime_rule_id",
    "price_regime_conflict_flag", "price_regime_review_flag", "price_regime_unclassified_flag", "price_regime_version",
    "exact_duplicate_flag", "exact_duplicate_excess_flag", "same_month_duplicate_group_size",
    "same_month_duplicate_flag", "same_month_duplicate_excess_flag", "probable_duplicate_cluster_id",
    "deduplication_confidence", "deduplication_rule_version", "supply_keep_flag",
    "cross_month_repeat_row_count", "cross_month_repeat_month_count", "cross_month_repeat_flag",
    "cross_month_first_month", "cross_month_last_month",
    "effective_building_area_sqm", "effective_land_area_sqm", "effective_land_area_source",
    "primary_area_required_flag", "area_manual_review_flag", "primary_area_missing_flag",
    "primary_area_nonpositive_flag", "primary_area_outlier_flag", "building_area_outlier_flag",
    "land_area_outlier_flag", "sale_price_per_sqm_outlier_flag", "monthly_rent_outlier_flag",
    "deposit_outlier_flag", "outlier_area_flag", "outlier_price_flag", "outlier_year_flag", "outlier_rule_version",
    "sale_price_analysis_eligible_flag", "rent_component_analysis_eligible_flag",
    "sale_total_price_analysis_eligible_flag", "sale_price_per_sqm_final_eligible_flag", "rent_final_eligible_flag",
    "supply_analysis_eligible_flag", "geo_aggregate_map_eligible_flag", "map_analysis_eligible_flag",
    "sale_price_final_toman", "sale_price_per_sqm_final_toman", "monthly_rent_final_toman", "deposit_final_toman",
    "rent_equivalent_monthly_low_toman", "rent_equivalent_monthly_base_toman", "rent_equivalent_monthly_high_toman",
    "rent_equivalent_deposit_low_toman", "rent_equivalent_deposit_base_toman", "rent_equivalent_deposit_high_toman",
    "rent_equivalent_rule_version", "price_unit", "price_observation_type", "quality_issue_count",
    "record_quality_review_flag", "quality_rule_version", "silver_master_version",
]


def show_progress(percent: int, label: str, *, final: bool = False) -> None:
    pct = max(0, min(100, int(percent)))
    filled = int(PROGRESS_WIDTH * pct / 100)
    bar = "#" * filled + "-" * (PROGRESS_WIDTH - filled)
    print(f"\rM2 progress [{bar}] {pct:3d}% complete | {100-pct:3d}% remaining | {label[:44]:44s}", end="\n" if final else "", flush=True)


def _csv_has_fail(path: Path) -> bool:
    if not path.exists():
        return True
    frame = pl.read_csv(path, infer_schema_length=1000)
    if "status" not in frame.columns:
        return False
    return not frame.filter(pl.col("status").cast(pl.String).str.to_uppercase() == "FAIL").is_empty()


def _raw_scan(raw_path: Path, max_rows: int | None = None) -> tuple[pl.LazyFrame, list[str]]:
    scan = pl.scan_csv(
        raw_path,
        has_header=True,
        infer_schema=False,
        try_parse_dates=False,
        ignore_errors=False,
        truncate_ragged_lines=False,
        low_memory=True,
        rechunk=False,
    )
    columns = scan.collect_schema().names()
    scan = scan.select([pl.col(column).cast(pl.String) for column in columns])
    if max_rows is not None:
        scan = scan.head(max_rows)
    return scan, columns


def build_candidate(raw_path: Path, candidate_path: Path, max_rows: int | None = None) -> tuple[int, list[str]]:
    scan, raw_columns = _raw_scan(raw_path, max_rows)
    frame = apply_structured_standardization(scan)
    frame = add_normalized_text_columns(frame)
    frame = apply_price_regimes(frame)
    frame = add_duplicate_features(frame)
    frame = apply_outlier_rules(frame)
    frame = apply_final_metrics(frame)
    available = set(frame.collect_schema().names())
    missing_derived = [column for column in DERIVED_COLUMNS if column not in available]
    if missing_derived:
        raise RuntimeError(f"Planned Silver columns were not created: {missing_derived}")
    final_columns = [*raw_columns, *[column for column in DERIVED_COLUMNS if column not in raw_columns]]
    atomic_sink_parquet(frame.select(final_columns), candidate_path)
    row_count = int(pl.scan_parquet(candidate_path).select(pl.len()).collect(engine="streaming").item())
    return row_count, final_columns


def _derived_metadata(column: str, dtype: str) -> dict[str, str]:
    unit = "none"
    if column.endswith("_toman"):
        unit = "toman"
    elif column.endswith("_sqm"):
        unit = "sqm"
    elif column in {"latitude", "longitude"}:
        unit = "decimal_degree"
    elif column.endswith("_years"):
        unit = "year"
    elif column.endswith("_rate"):
        unit = "ratio"
    elif column.endswith("_count") or column.endswith("_rows") or column.endswith("_num"):
        unit = "count"

    if column.endswith("_flag") or dtype == "Boolean":
        allowed = "true | false"
        missing = "Should not be missing after the M2 canonical transformation; missing indicates a pipeline issue."
    elif column.endswith("_version"):
        allowed = "versioned project rule identifier"
        missing = "Should not be missing for published Silver rows."
    else:
        allowed = "documented by source/domain rule; null when unavailable or not applicable"
        missing = "Null means unavailable, unknown, invalid for the metric, or not applicable under the field contract."

    explicit_meanings = {
        "source_row_id": "Stable 1-based row identifier assigned from Raw source order for audit joins.",
        "analysis_month": "Parsed listing month used for the project analysis window.",
        "property_family": "Shared canonical property family derived from cat3_slug.",
        "primary_area_sqm": "Canonical denominator area used only for compatible property categories.",
        "price_regime": "Canonical sale/rent/service price-regime classification.",
        "probable_duplicate_cluster_id": "Deterministic cluster id for high/medium probable duplicate candidates.",
        "deduplication_confidence": "Duplicate/ER confidence: exact, high, medium, or none.",
        "supply_keep_flag": "Whether the row is retained in canonical listing-supply counts after conservative same-month deduplication.",
        "sale_price_per_sqm_final_toman": "Final eligible sale asking price per square meter in assumed toman.",
        "rent_equivalent_monthly_base_toman": "Base-scenario monthly equivalent asking rent combining rent and deposit.",
        "price_unit": "Operational currency-unit status; source remains unconfirmed.",
        "price_observation_type": "Asking-price observation label; not transaction price.",
        "silver_master_version": "Version identifier of the published canonical Silver Master schema/rules.",
    }
    if column.endswith("_bool"):
        source = column.removesuffix("_bool")
        meaning = f"Typed Boolean interpretation of Raw field {source}; unknown is preserved as null."
    elif column.endswith("_toman") and column not in explicit_meanings:
        meaning = f"Canonical/typed monetary field {column} in the project's assumed toman unit."
    elif column.endswith("_flag") and column not in explicit_meanings:
        meaning = f"Boolean quality, review, censoring, or analytical-eligibility indicator: {column}."
    else:
        meaning = explicit_meanings.get(column, f"Canonical M2 derived field: {column}.")

    return {
        "column_name": column,
        "business_meaning": meaning,
        "raw_dtype": "N/A (derived)",
        "expected_dtype": dtype,
        "unit": unit,
        "allowed_values": allowed,
        "missing_meaning": missing,
        "cleaning_rule": "Created by the canonical Milestone 2 module responsible for this field; Raw source values remain preserved separately.",
        "quality_risk": "Interpret only within its documented applicability and rule version; do not treat null as zero/False unless explicitly defined.",
        "owner": "Milestone 2",
    }


def reconcile_dictionary_and_contract(silver_path: Path, raw_columns: list[str]) -> None:
    dictionary_path = DOCS_DIR / "data_dictionary.csv"
    existing: dict[str, dict[str, str]] = {}
    if dictionary_path.exists():
        with dictionary_path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("column_name"):
                    existing[row["column_name"]] = row

    schema = pl.read_parquet_schema(silver_path)
    field_order = [
        "column_name", "business_meaning", "raw_dtype", "expected_dtype", "unit",
        "allowed_values", "missing_meaning", "cleaning_rule", "quality_risk", "owner",
    ]
    rows: list[dict[str, str]] = []
    for column, dtype in schema.items():
        if column in existing and column in raw_columns:
            row = {key: str(existing[column].get(key, "")) for key in field_order}
            row["expected_dtype"] = str(dtype)
            rows.append(row)
        else:
            rows.append(_derived_metadata(column, str(dtype)))

    dictionary_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = dictionary_path.with_name(f".{dictionary_path.name}.tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_order)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, dictionary_path)

    contract = {
        "contract_version": "silver-master-contract-v1",
        "dataset": "silver_master",
        "stage": "milestone_2_canonical_release",
        "expected_column_count": len(rows),
        "primary_key": "source_row_id",
        "global_assumptions": {
            "price_unit": price_unit(),
            "price_observation_type": price_observation_type(),
            "raw_columns_preserved": True,
            "no_blanket_zero_imputation": True,
            "exact_coordinates_internal_only": True,
        },
        "required_column_metadata": field_order,
        "columns": rows,
    }
    atomic_write_text(yaml.safe_dump(contract, allow_unicode=True, sort_keys=False), CONFIG_DIR / "data_contract.yaml")


def run_full_build(raw_path: Path, final_path: Path, overwrite: bool) -> int:
    if final_path.exists() and not overwrite:
        raise FileExistsError(f"Silver Master already exists: {final_path}. Use --overwrite only after review.")
    candidate = final_path.with_name(f".{final_path.stem}.candidate.parquet")
    candidate.unlink(missing_ok=True)
    started = perf_counter()
    stage_started = started
    performance: list[dict[str, object]] = []

    show_progress(0, "starting M2 canonical build")
    raw_scan, raw_columns = _raw_scan(raw_path)
    raw_rows = int(raw_scan.select(pl.len()).collect(engine="streaming").item())
    now = perf_counter(); performance.append({"stage": "raw_validation", "elapsed_seconds": round(now-stage_started, 4)}); stage_started = now
    show_progress(8, f"Raw source validated ({raw_rows:,} rows)")

    candidate_rows, final_columns = build_candidate(raw_path, candidate)
    now = perf_counter(); performance.append({"stage": "candidate_build", "elapsed_seconds": round(now-stage_started, 4)}); stage_started = now
    show_progress(62, f"Silver candidate built ({candidate_rows:,} rows)")

    build_checks = pl.DataFrame([
        {"check": "row_count_preserved", "actual": str(candidate_rows), "expected": str(raw_rows), "status": "PASS" if candidate_rows == raw_rows else "FAIL"},
        {"check": "raw_columns_preserved", "actual": str(len([c for c in raw_columns if c in final_columns])), "expected": str(len(raw_columns)), "status": "PASS" if all(c in final_columns for c in raw_columns) else "FAIL"},
        {"check": "silver_column_count", "actual": str(len(final_columns)), "expected": "purpose-driven (not fixed)", "status": "PASS"},
    ])
    build_check_path = OUTPUTS_DIR / "qa" / "milestone_2" / "silver_build_checks.csv"
    atomic_write_csv(build_checks, build_check_path)
    if not build_checks.filter(pl.col("status") == "FAIL").is_empty():
        raise RuntimeError("Silver candidate failed build reconciliation; final file was not published.")

    standardization_path = write_standardization_summary(candidate)
    missingness_path = run_missingness_audit(candidate)
    duplicate_paths = write_duplicate_reports(candidate)
    currency_path = run_currency_validation(candidate)
    regime_path = write_price_regime_review(candidate)
    outlier_paths = run_outlier_validation(candidate)
    metric_path, metric_ready = run_final_metric_validation(candidate)
    now = perf_counter(); performance.append({"stage": "reports_and_validations", "elapsed_seconds": round(now-stage_started, 4)}); stage_started = now
    show_progress(84, "M2 task reports and validations complete")

    gate_path, gate_manifest, gate_ready = run_quality_gate(candidate)
    now = perf_counter(); performance.append({"stage": "quality_gate", "elapsed_seconds": round(now-stage_started, 4)}); stage_started = now
    blocking_fail = _csv_has_fail(currency_path) or not metric_ready or not gate_ready
    if blocking_fail:
        show_progress(92, "quality gate failed; candidate retained for review")
        raise RuntimeError(f"M2 quality gate failed. Review {relative_to_project(gate_path)}")

    final_path.parent.mkdir(parents=True, exist_ok=True)
    if final_path.exists():
        final_path.unlink()
    os.replace(candidate, final_path)
    reconcile_dictionary_and_contract(final_path, raw_columns)
    now = perf_counter(); performance.append({"stage": "publish_and_documentation", "elapsed_seconds": round(now-stage_started, 4)})
    performance.append({"stage": "total", "elapsed_seconds": round(now-started, 4)})
    atomic_write_csv(pl.DataFrame(performance), OUTPUTS_DIR / "qa" / "milestone_2" / "performance_log.csv")
    show_progress(100, f"Silver Master published in {now-started:.1f}s", final=True)
    print(f"silver_master: {relative_to_project(final_path)}")
    print(f"columns: {len(final_columns)} (engineered fields)")
    print(f"quality_gate: {relative_to_project(gate_manifest)}")
    return 0


def run_smoke(raw_path: Path, rows: int) -> int:
    smoke_dir = OUTPUTS_DIR / "qa" / "milestone_2" / "smoke"
    smoke_path = smoke_dir / ".silver_smoke.parquet"
    smoke_path.unlink(missing_ok=True)
    show_progress(0, f"starting {rows:,}-row M2 smoke test")
    row_count, columns = build_candidate(raw_path, smoke_path, max_rows=rows)
    schema = pl.read_parquet_schema(smoke_path)
    smoke_path.unlink(missing_ok=True)
    show_progress(100, f"smoke test passed: {row_count:,} rows / {len(columns)} columns", final=True)
    print(f"schema_fields={len(schema)}")
    print("No smoke dataset was retained.")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the canonical M2 Silver Master from Raw source.")
    parser.add_argument("--raw", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--smoke-rows", type=int, default=None, help="Transient development test only; no dataset is retained.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_path = (args.raw or configured_path("raw_source")).resolve()
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw source not found: {raw_path}")
    if args.smoke_rows:
        raise SystemExit(run_smoke(raw_path, args.smoke_rows))
    raise SystemExit(run_full_build(raw_path, configured_path("silver_master"), args.overwrite))


if __name__ == "__main__":
    main()
