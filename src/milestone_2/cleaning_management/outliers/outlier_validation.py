from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from src.common.config import setting
from src.common.io_utils import atomic_write_csv
from src.common.paths import OUTPUTS_DIR

CANONICAL_FLAGS = [
    "primary_area_outlier_flag", "building_area_outlier_flag", "land_area_outlier_flag",
    "sale_price_per_sqm_outlier_flag", "monthly_rent_outlier_flag", "deposit_outlier_flag",
    "outlier_area_flag", "outlier_price_flag", "outlier_year_flag",
]


def _global_sensitivity(scan: pl.LazyFrame, value_column: str, canonical_flag: str, label: str) -> dict[str, object]:
    multiplier = float(setting("milestone_2", "outliers", "sensitivity_iqr_multiplier", default=2.0))
    base = scan.filter(
        pl.col("is_core_analysis_period") & pl.col("supply_keep_flag")
        & pl.col(value_column).is_not_null() & (pl.col(value_column) > 0)
    )
    stats = base.select([
        pl.len().alias("n"),
        pl.col(value_column).quantile(0.25).alias("q1"),
        pl.col(value_column).quantile(0.75).alias("q3"),
        pl.col(value_column).log().median().alias("log_median"),
    ]).collect(engine="streaming").row(0, named=True)
    n = int(stats["n"] or 0)
    if n == 0 or stats["q1"] is None or stats["q3"] is None:
        return {"metric": label, "evaluated_rows": n, "canonical_outlier_rows": 0, "iqr2_outlier_rows": 0, "log_mad_outlier_rows": 0, "status": "REVIEW"}
    iqr = float(stats["q3"] - stats["q1"])
    lower = max(0.0, float(stats["q1"]) - multiplier * iqr)
    upper = float(stats["q3"]) + multiplier * iqr
    log_median = float(stats["log_median"])
    mad_row = base.select((pl.col(value_column).log() - log_median).abs().median().alias("mad")).collect(engine="streaming").row(0, named=True)
    mad = float(mad_row["mad"] or 0.0)
    log_mad_flag = pl.lit(False) if mad <= 0 else (((pl.col(value_column).log() - log_median).abs() / (1.4826 * mad)) > 3.5)
    counts = base.select([
        pl.col(canonical_flag).sum().alias("canonical"),
        ((pl.col(value_column) < lower) | (pl.col(value_column) > upper)).sum().alias("iqr2"),
        log_mad_flag.fill_null(False).sum().alias("log_mad"),
    ]).collect(engine="streaming").row(0, named=True)
    return {
        "metric": label,
        "evaluated_rows": n,
        "canonical_outlier_rows": int(counts["canonical"] or 0),
        "iqr2_outlier_rows": int(counts["iqr2"] or 0),
        "log_mad_outlier_rows": int(counts["log_mad"] or 0),
        "status": "PASS",
    }


def run_outlier_validation(input_path: Path, output_dir: Path | None = None) -> tuple[Path, Path]:
    output_dir = output_dir or OUTPUTS_DIR / "tables" / "milestone_2" / "outliers"
    scan = pl.scan_parquet(input_path)
    stats = scan.select([pl.len().alias("row_count"), *[pl.col(flag).sum().alias(flag) for flag in CANONICAL_FLAGS]]).collect(engine="streaming").row(0, named=True)
    summary_rows = [{"metric": key, "row_count": int(value or 0), "status": "PASS", "notes": "Flag-only policy; rows are retained in Silver."} for key, value in stats.items() if key != "row_count"]
    summary_rows.insert(0, {"metric": "dataset_rows", "row_count": int(stats["row_count"]), "status": "PASS", "notes": "No outlier rule deletes source rows."})
    summary_path = output_dir / "outlier_summary.csv"
    atomic_write_csv(pl.DataFrame(summary_rows), summary_path)

    sensitivity = pl.DataFrame([
        _global_sensitivity(scan, "price_per_sqm_raw_toman", "sale_price_per_sqm_outlier_flag", "sale_price_per_sqm"),
        _global_sensitivity(scan, "monthly_rent_clean_toman", "monthly_rent_outlier_flag", "monthly_rent"),
        _global_sensitivity(scan, "deposit_clean_toman", "deposit_outlier_flag", "deposit"),
    ])
    sensitivity_path = output_dir / "outlier_sensitivity.csv"
    atomic_write_csv(sensitivity, sensitivity_path)
    return summary_path, sensitivity_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Write M2 outlier summary and sensitivity evidence.")
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    for path in run_outlier_validation(args.input):
        print(path)


if __name__ == "__main__":
    main()
