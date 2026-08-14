from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from src.common.config import setting
from src.common.io_utils import atomic_write_csv
from src.common.paths import OUTPUTS_DIR


def _float_mismatch(actual: str, expected: pl.Expr, tolerance: float = 1e-6) -> pl.Expr:
    return (
        pl.col(actual).is_not_null()
        & expected.is_not_null()
        & ((pl.col(actual).cast(pl.Float64) - expected.cast(pl.Float64)).abs() > tolerance)
    ).fill_null(False)


def run_final_metric_validation(input_path: Path, output_path: Path | None = None) -> tuple[Path, bool]:
    output_path = output_path or OUTPUTS_DIR / "tables" / "milestone_2" / "final_metrics" / "final_metric_summary.csv"
    scan = pl.scan_parquet(input_path)
    low = int(setting("analysis", "rent_equivalence_k_toman", "low", default=25_000))
    base = int(setting("analysis", "rent_equivalence_k_toman", "base", default=30_000))
    high = int(setting("analysis", "rent_equivalence_k_toman", "high", default=35_000))

    expected_psm = pl.col("sale_price_clean_toman").cast(pl.Float64) / pl.col("primary_area_sqm")
    expected_monthly = {
        "low": pl.col("monthly_rent_clean_toman").cast(pl.Float64) + pl.col("deposit_clean_toman").cast(pl.Float64) * (low / 1_000_000.0),
        "base": pl.col("monthly_rent_clean_toman").cast(pl.Float64) + pl.col("deposit_clean_toman").cast(pl.Float64) * (base / 1_000_000.0),
        "high": pl.col("monthly_rent_clean_toman").cast(pl.Float64) + pl.col("deposit_clean_toman").cast(pl.Float64) * (high / 1_000_000.0),
    }
    expected_deposit = {
        "low": pl.col("deposit_clean_toman").cast(pl.Float64) + pl.col("monthly_rent_clean_toman").cast(pl.Float64) / low * 1_000_000.0,
        "base": pl.col("deposit_clean_toman").cast(pl.Float64) + pl.col("monthly_rent_clean_toman").cast(pl.Float64) / base * 1_000_000.0,
        "high": pl.col("deposit_clean_toman").cast(pl.Float64) + pl.col("monthly_rent_clean_toman").cast(pl.Float64) / high * 1_000_000.0,
    }

    expressions = [
        pl.len().alias("row_count"),
        pl.col("sale_price_per_sqm_final_eligible_flag").sum().alias("sale_psm_eligible_rows"),
        pl.col("sale_price_per_sqm_final_toman").is_not_null().sum().alias("sale_psm_populated_rows"),
        (_float_mismatch("sale_price_per_sqm_final_toman", expected_psm) & pl.col("sale_price_per_sqm_final_eligible_flag")).sum().alias("sale_psm_mismatch_rows"),
        pl.col("rent_final_eligible_flag").sum().alias("rent_eligible_rows"),
        pl.col("rent_equivalent_monthly_base_toman").is_not_null().sum().alias("rent_equivalent_populated_rows"),
    ]
    for name in ["low", "base", "high"]:
        expressions.append((_float_mismatch(f"rent_equivalent_monthly_{name}_toman", expected_monthly[name].round(0)) & pl.col("rent_final_eligible_flag")).sum().alias(f"monthly_{name}_mismatch_rows"))
        expressions.append((_float_mismatch(f"rent_equivalent_deposit_{name}_toman", expected_deposit[name].round(0)) & pl.col("rent_final_eligible_flag")).sum().alias(f"deposit_{name}_mismatch_rows"))
    values = scan.select(expressions).collect(engine="streaming").row(0, named=True)

    rows = []
    rows.append({
        "metric": "sale_price_per_sqm",
        "eligible_rows": int(values["sale_psm_eligible_rows"] or 0),
        "populated_rows": int(values["sale_psm_populated_rows"] or 0),
        "mismatch_rows": int(values["sale_psm_mismatch_rows"] or 0),
        "status": "PASS" if int(values["sale_psm_mismatch_rows"] or 0) == 0 and values["sale_psm_eligible_rows"] == values["sale_psm_populated_rows"] else "FAIL",
        "notes": "Canonical asking sale price per sqm = fixed positive sale asking price / valid primary area.",
    })
    rent_mismatches = sum(int(values[f"monthly_{name}_mismatch_rows"] or 0) + int(values[f"deposit_{name}_mismatch_rows"] or 0) for name in ["low", "base", "high"])
    rows.append({
        "metric": "rent_equivalent_sensitivity",
        "eligible_rows": int(values["rent_eligible_rows"] or 0),
        "populated_rows": int(values["rent_equivalent_populated_rows"] or 0),
        "mismatch_rows": rent_mismatches,
        "status": "PASS" if rent_mismatches == 0 and values["rent_eligible_rows"] == values["rent_equivalent_populated_rows"] else "FAIL",
        "notes": f"Monthly/deposit-equivalent formulas validated for k={low:,}/{base:,}/{high:,} toman.",
    })
    result = pl.DataFrame(rows)
    atomic_write_csv(result, output_path)
    ready = result.filter(pl.col("status") == "FAIL").is_empty()
    return output_path, ready


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate canonical M2 price metrics.")
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    path, ready = run_final_metric_validation(args.input)
    print(path)
    raise SystemExit(0 if ready else 1)


if __name__ == "__main__":
    main()
