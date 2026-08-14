from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from src.common.config import price_observation_type, price_unit, setting
from src.common.io_utils import atomic_write_csv, atomic_write_json
from src.common.paths import OUTPUTS_DIR

REQUIRED_COLUMNS = {
    "source_row_id", "analysis_month", "price_regime", "price_regime_rule_id",
    "exact_duplicate_flag", "probable_duplicate_cluster_id", "deduplication_confidence",
    "supply_keep_flag", "outlier_price_flag", "outlier_area_flag", "outlier_year_flag",
    "sale_price_per_sqm_final_toman", "rent_equivalent_monthly_base_toman",
    "supply_analysis_eligible_flag", "map_analysis_eligible_flag", "price_unit",
    "price_observation_type", "silver_master_version",
}


def _row(check: str, actual: object, expected: object, status: str, *, critical: bool, notes: str = "") -> dict[str, object]:
    return {
        "check": check,
        "actual": str(actual),
        "expected": str(expected),
        "status": status,
        "critical": critical,
        "notes": notes,
    }


def _minimum_n_sensitivity(scan: pl.LazyFrame) -> pl.DataFrame:
    thresholds = [int(x) for x in setting("analysis", "minimum_valid_listings", "sensitivity", default=[10, 20, 30, 50, 100])]
    grouped = (
        scan.filter(pl.col("sale_price_per_sqm_final_eligible_flag"))
        .group_by(["city_slug", "neighborhood_slug"])
        .agg(pl.len().alias("valid_listing_count"))
        .collect(engine="streaming")
    )
    rows = []
    for threshold in thresholds:
        eligible_groups = grouped.filter(pl.col("valid_listing_count") >= threshold).height
        eligible_rows = int(grouped.filter(pl.col("valid_listing_count") >= threshold).get_column("valid_listing_count").sum() or 0)
        rows.append({
            "minimum_valid_listings": threshold,
            "eligible_city_neighborhood_groups": eligible_groups,
            "eligible_sale_listing_rows": eligible_rows,
            "is_default": threshold == int(setting("analysis", "minimum_valid_listings", "default", default=30)),
        })
    return pl.DataFrame(rows)


def _data_quality_action_table(scan: pl.LazyFrame) -> pl.DataFrame:
    n = int(scan.select(pl.len()).collect(engine="streaming").item())
    counts = scan.select([
        (pl.col("type_parse_error_count") > 0).sum().alias("parse"),
        pl.col("price_regime_review_flag").sum().alias("regime"),
        (~pl.col("supply_keep_flag")).sum().alias("duplicate_excess"),
        pl.col("outlier_price_flag").sum().alias("outlier_price"),
        pl.col("outlier_area_flag").sum().alias("outlier_area"),
        pl.col("outlier_year_flag").sum().alias("outlier_year"),
        (pl.col("coordinate_pair_present") & ~pl.col("geo_country_valid")).sum().alias("geo"),
    ]).collect(engine="streaming").row(0, named=True)
    rows = [
        ("multiple", "parse_failure", counts["parse"], "Raw text could not be safely parsed to the canonical type.", "Keep Raw value, flag the record, and exclude only from analyses needing that parsed field.", "Avoid inventing values during type conversion.", "Can reduce eligible sample sizes for price/model analysis."),
        ("price fields", "price_regime_review", counts["regime"], "Price components are negotiable, incomplete, conflicting, or outside a canonical property regime.", "Retain and label; do not coerce into a numeric regime.", "Regime errors mix non-comparable prices.", "Can bias sale/rent metrics if included."),
        ("listing identity", "duplicate_excess", counts["duplicate_excess"], "Same listing/entity is repeated within the same month with high confidence.", "Retain source rows but set supply_keep_flag=False for excess exact/high-confidence rows.", "Supply KPIs should not count duplicated platform listings twice.", "Can inflate listing-supply measures."),
        ("price metrics", "outlier_price", counts["outlier_price"], "Value is extreme under the predeclared grouped-IQR policy.", "Flag only; exclude from canonical comparable-price metrics, never delete from Silver.", "Extreme values may be valid but distort robust market summaries.", "Can distort medians/models if filters are ignored."),
        ("area fields", "outlier_area", counts["outlier_area"], "Area is nonpositive or extreme for its compatible property group.", "Flag only and exclude from price-per-sqm eligibility when relevant.", "Area is the denominator of unit-price metrics.", "Can create invalid or highly distorted price-per-sqm values."),
        ("construction_year", "outlier_or_censored_year", counts["outlier_year"], "Year is pre-1370 censored or numerically outside the listing-date domain.", "Preserve censor flag; keep exact age null when year is not exactly known.", "A censored category is not an exact construction year.", "Can bias age effects if treated as a numeric year."),
        ("coordinates", "coordinate_quality", counts["geo"], "Coordinate pair is present but outside the broad Iran validation window.", "Retain internally, flag, and exclude from aggregate map eligibility pending M3 spatial QA.", "Invalid coordinates should not enter maps.", "Can place listings in the wrong geography."),
    ]
    return pl.DataFrame([
        {
            "column": column,
            "issue": issue,
            "affected_rate": (int(count or 0) / n if n else None),
            "missing_error_meaning": meaning,
            "action": action,
            "reason": reason,
            "possible_effect": effect,
        }
        for column, issue, count, meaning, action, reason, effect in rows
    ])


def run_quality_gate(input_path: Path) -> tuple[Path, Path, bool]:
    table_dir = OUTPUTS_DIR / "tables" / "milestone_2" / "quality_gate"
    qa_dir = OUTPUTS_DIR / "qa" / "milestone_2" / "quality_gate"
    summary_path = table_dir / "quality_gate_summary.csv"
    manifest_path = qa_dir / "quality_gate_summary.json"
    scan = pl.scan_parquet(input_path)
    schema = set(scan.collect_schema().names())
    missing = sorted(REQUIRED_COLUMNS - schema)

    checks: list[dict[str, object]] = []
    checks.append(_row("required_silver_columns", len(missing), 0, "PASS" if not missing else "FAIL", critical=True, notes=", ".join(missing)))
    if missing:
        result = pl.DataFrame(checks)
        atomic_write_csv(result, summary_path)
        payload = {"overall_status": "FAIL", "m3_entry_ready": False, "critical_failures": 1, "reviews": 0}
        atomic_write_json(payload, manifest_path)
        return summary_path, manifest_path, False

    values = scan.select([
        pl.len().alias("row_count"),
        pl.col("source_row_id").n_unique().alias("unique_ids"),
        pl.col("price_regime").is_null().sum().alias("null_regime"),
        pl.col("price_unit").is_null().sum().alias("null_price_unit"),
        pl.col("price_observation_type").is_null().sum().alias("null_observation"),
        pl.col("silver_master_version").is_null().sum().alias("null_version"),
        pl.col("sale_price_per_sqm_final_toman").is_infinite().sum().alias("nonfinite_sale_psm"),
        pl.col("rent_equivalent_monthly_base_toman").cast(pl.Float64).is_infinite().sum().alias("nonfinite_rent_equiv"),
        pl.col("sale_price_per_sqm_final_eligible_flag").sum().alias("sale_psm_eligible"),
        pl.col("rent_final_eligible_flag").sum().alias("rent_eligible"),
        pl.col("supply_analysis_eligible_flag").sum().alias("supply_eligible"),
        pl.col("map_analysis_eligible_flag").sum().alias("map_eligible"),
        (pl.col("exact_duplicate_excess_flag") & pl.col("supply_keep_flag")).sum().alias("exact_excess_kept"),
        (pl.col("same_month_duplicate_excess_flag") & pl.col("supply_keep_flag")).sum().alias("high_excess_kept"),
        pl.col("price_unit").drop_nulls().n_unique().alias("price_unit_domain_count"),
        pl.col("price_observation_type").drop_nulls().n_unique().alias("observation_domain_count"),
    ]).collect(engine="streaming").row(0, named=True)

    n = int(values["row_count"])
    checks += [
        _row("source_row_id_unique", values["unique_ids"], n, "PASS" if values["unique_ids"] == n else "FAIL", critical=True),
        _row("price_regime_null_rows", values["null_regime"], 0, "PASS" if values["null_regime"] == 0 else "FAIL", critical=True),
        _row("price_unit_null_rows", values["null_price_unit"], 0, "PASS" if values["null_price_unit"] == 0 else "FAIL", critical=True),
        _row("price_observation_type_null_rows", values["null_observation"], 0, "PASS" if values["null_observation"] == 0 else "FAIL", critical=True),
        _row("silver_master_version_null_rows", values["null_version"], 0, "PASS" if values["null_version"] == 0 else "FAIL", critical=True),
        _row("nonfinite_sale_psm_rows", values["nonfinite_sale_psm"], 0, "PASS" if values["nonfinite_sale_psm"] == 0 else "FAIL", critical=True),
        _row("nonfinite_rent_equivalent_rows", values["nonfinite_rent_equiv"], 0, "PASS" if values["nonfinite_rent_equiv"] == 0 else "FAIL", critical=True),
        _row("exact_duplicate_excess_rows_kept", values["exact_excess_kept"], 0, "PASS" if values["exact_excess_kept"] == 0 else "FAIL", critical=True),
        _row("high_probable_duplicate_excess_rows_kept", values["high_excess_kept"], 0, "PASS" if values["high_excess_kept"] == 0 else "FAIL", critical=True),
        _row("sale_psm_eligible_rows", values["sale_psm_eligible"], ">0", "PASS" if int(values["sale_psm_eligible"] or 0) > 0 else "FAIL", critical=True),
        _row("rent_eligible_rows", values["rent_eligible"], ">0", "PASS" if int(values["rent_eligible"] or 0) > 0 else "FAIL", critical=True),
        _row("supply_eligible_rows", values["supply_eligible"], ">0", "PASS" if int(values["supply_eligible"] or 0) > 0 else "FAIL", critical=True),
        _row("map_eligible_rows", values["map_eligible"], ">0", "PASS" if int(values["map_eligible"] or 0) > 0 else "REVIEW", critical=False, notes="Advanced spatial QA is owned by M3 §21."),
        _row("currency_source_confirmation", price_unit(), "toman_assumed_unconfirmed", "REVIEW" if price_unit() == "toman_assumed_unconfirmed" else "PASS", critical=False, notes="Documented project limitation; no undocumented factor-of-ten conversion is applied."),
        _row("price_observation_contract", price_observation_type(), "asking_price", "PASS" if price_observation_type() == "asking_price" else "FAIL", critical=True),
    ]

    result = pl.DataFrame(checks)
    atomic_write_csv(result, summary_path)
    sensitivity = _minimum_n_sensitivity(scan)
    atomic_write_csv(sensitivity, qa_dir / "minimum_n_sensitivity.csv")
    atomic_write_csv(_data_quality_action_table(scan), OUTPUTS_DIR / "tables" / "milestone_2" / "data_quality_action_table.csv")

    critical_failures = result.filter(pl.col("critical") & (pl.col("status") == "FAIL")).height
    reviews = result.filter(pl.col("status") == "REVIEW").height
    overall = "FAIL" if critical_failures else ("REVIEW" if reviews else "PASS")
    ready = critical_failures == 0
    payload = {
        "overall_status": overall,
        "m3_entry_ready": ready,
        "critical_failures": critical_failures,
        "review_count": reviews,
        "row_count": n,
        "minimum_valid_listings_default": int(setting("analysis", "minimum_valid_listings", "default", default=30)),
    }
    atomic_write_json(payload, manifest_path)
    return summary_path, manifest_path, ready


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the canonical M2 quality gate.")
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    summary, manifest, ready = run_quality_gate(args.input)
    print(summary)
    print(manifest)
    raise SystemExit(0 if ready else 1)


if __name__ == "__main__":
    main()
