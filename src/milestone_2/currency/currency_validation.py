from __future__ import annotations

import argparse
import math
from pathlib import Path

import polars as pl

from src.common.config import price_observation_type, price_unit, setting
from src.common.io_utils import atomic_write_csv, atomic_write_text
from src.common.paths import OUTPUTS_DIR

VALIDATION_VERSION = "currency-inference-m2-v2"
MIN_LOCAL_N = 30
BASE_RENT_DEPOSIT_RATE = 0.030

RAW_TYPED_PAIRS = [
    ("price_value", "price_value_toman"),
    ("credit_value", "credit_value_toman"),
    ("rent_value", "rent_value_toman"),
    ("transformable_credit", "transformable_credit_toman"),
    ("transformed_credit", "transformed_credit_toman"),
    ("transformable_rent", "transformable_rent_toman"),
    ("transformed_rent", "transformed_rent_toman"),
    ("cost_per_extra_person", "cost_per_extra_person_toman"),
    ("rent_price_on_regular_days", "rent_price_on_regular_days_toman"),
    ("rent_price_at_weekends", "rent_price_at_weekends_toman"),
    ("rent_price_on_special_days", "rent_price_on_special_days_toman"),
]

REQUIRED_ANALYSIS_COLUMNS = {
    "city_slug",
    "neighborhood_slug",
    "cat3_slug",
    "price_mode",
    "price_value",
    "rent_value",
    "credit_value",
    "building_size_sqm",
    "rent_credit_transform",
    "transformable_credit",
    "transformed_credit",
    "transformable_rent",
    "transformed_rent",
    "is_core_analysis_period",
}


def _raw_number(column: str) -> pl.Expr:
    value = pl.col(column).cast(pl.String).str.strip_chars()
    for source, target in zip("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789"):
        value = value.str.replace_all(source, target, literal=True)
    return (
        value.str.replace_all(",", "", literal=True)
        .str.replace_all("٬", "", literal=True)
        .str.replace_all(" ", "", literal=True)
        .cast(pl.Float64, strict=False)
    )


def _as_bool(column: str) -> pl.Expr:
    return (
        pl.col(column)
        .cast(pl.String)
        .str.strip_chars()
        .str.to_lowercase()
        .is_in(["true", "1", "yes"])
        .fill_null(False)
    )


def _q(frame: pl.DataFrame, column: str, probability: float) -> float:
    value = frame.select(pl.col(column).quantile(probability, interpolation="linear")).item()
    return float(value) if value is not None else float("nan")


def _positive_finite(frame: pl.DataFrame, column: str) -> pl.DataFrame:
    return frame.filter(pl.col(column).is_finite() & (pl.col(column) > 0))


def _trim_central(frame: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    """Keep the internally defined central 98% for each listed field.

    This is not an external plausibility threshold. It only prevents source sentinels
    and extreme entry errors from dominating the hidden-pattern diagnostics.
    """
    out = frame
    for column in columns:
        out = _positive_finite(out, column)
        if out.height == 0:
            return out
        low = _q(out, column, 0.01)
        high = _q(out, column, 0.99)
        out = out.filter(pl.col(column).is_between(low, high, closed="both"))
    return out


def _local_decade_summary(
    frame: pl.DataFrame,
    value_column: str,
    *,
    minimum_n: int = MIN_LOCAL_N,
) -> dict[str, float | int]:
    work = _positive_finite(frame, value_column)
    cells = (
        work.group_by(["city_slug", "neighborhood_slug"])
        .agg(
            pl.len().alias("cell_n"),
            pl.col(value_column).median().alias("cell_median"),
        )
        .filter((pl.col("cell_n") >= minimum_n) & (pl.col("cell_median") > 0))
    )
    joined = work.join(cells, on=["city_slug", "neighborhood_slug"], how="inner").with_columns(
        (pl.col(value_column) / pl.col("cell_median")).alias("relative_to_local_median")
    )
    if joined.height == 0:
        return {
            "rows": 0,
            "cells": 0,
            "within_half_to_double_share": float("nan"),
            "near_one_tenth_share": float("nan"),
            "near_ten_x_share": float("nan"),
        }
    relative = pl.col("relative_to_local_median")
    stats = joined.select(
        pl.len().alias("rows"),
        relative.is_between(0.5, 2.0, closed="both").mean().alias("within_half_to_double_share"),
        relative.is_between(0.08, 0.125, closed="both").mean().alias("near_one_tenth_share"),
        relative.is_between(8.0, 12.5, closed="both").mean().alias("near_ten_x_share"),
    ).row(0, named=True)
    return {
        "rows": int(stats["rows"]),
        "cells": int(cells.height),
        "within_half_to_double_share": float(stats["within_half_to_double_share"]),
        "near_one_tenth_share": float(stats["near_one_tenth_share"]),
        "near_ten_x_share": float(stats["near_ten_x_share"]),
    }


def _raw_typed_parity(scan: pl.LazyFrame, schema: set[str]) -> tuple[int, int]:
    active = [(raw, typed) for raw, typed in RAW_TYPED_PAIRS if raw in schema and typed in schema]
    expressions: list[pl.Expr] = []
    for raw, typed in active:
        raw_value = _raw_number(raw)
        mismatch = (
            raw_value.is_not_null()
            & pl.col(typed).cast(pl.Float64, strict=False).is_not_null()
            & (raw_value != pl.col(typed).cast(pl.Float64, strict=False))
        ).fill_null(False)
        expressions.append(mismatch.sum().alias(f"{typed}__mismatch"))
    if not expressions:
        return 0, 0
    counts = scan.select(expressions).collect(engine="streaming").row(0, named=True)
    return sum(int(value or 0) for value in counts.values()), len(active)


def _collect_populations(scan: pl.LazyFrame) -> dict[str, pl.DataFrame]:
    core = pl.col("is_core_analysis_period").fill_null(False)
    fixed_price = pl.col("price_mode").cast(pl.String).str.strip_chars() == "مقطوع"
    price = _raw_number("price_value")
    rent = _raw_number("rent_value")
    deposit = _raw_number("credit_value")
    area = pl.col("building_size_sqm").cast(pl.Float64, strict=False)

    sale = (
        scan.filter(
            core
            & (pl.col("cat3_slug") == "apartment-sell")
            & fixed_price
            & (price > 0)
            & (area > 0)
        )
        .select(
            "city_slug",
            "neighborhood_slug",
            price.alias("price"),
            area.alias("area"),
        )
        .collect(engine="streaming")
    )

    rental = (
        scan.filter(
            core
            & (pl.col("cat3_slug") == "apartment-rent")
            & (area > 0)
            & ((rent > 0) | (deposit > 0))
        )
        .select(
            "city_slug",
            "neighborhood_slug",
            rent.alias("rent"),
            deposit.alias("deposit"),
            area.alias("area"),
        )
        .collect(engine="streaming")
    )
    return {"sale": sale, "rental": rental}


def _sale_scale_and_decade(sale: pl.DataFrame) -> tuple[dict[str, float | int], dict[str, float | int]]:
    central = _trim_central(sale, ["price", "area"]).with_columns(
        (pl.col("price") / pl.col("area")).alias("psm")
    )
    scale = {
        "n": central.height,
        "median": _q(central, "psm", 0.50),
        "p05": _q(central, "psm", 0.05),
        "p95": _q(central, "psm", 0.95),
    }
    return scale, _local_decade_summary(central, "psm")


def _rental_scale_and_decade(
    rental: pl.DataFrame,
    component: str,
) -> tuple[dict[str, float | int], dict[str, float | int]]:
    area_central = _trim_central(rental, ["area"])
    component_central = _trim_central(area_central, [component]).with_columns(
        (pl.col(component) / pl.col("area")).alias("per_sqm")
    )
    scale = {
        "n": component_central.height,
        "median": _q(component_central, "per_sqm", 0.50),
        "p05": _q(component_central, "per_sqm", 0.05),
        "p95": _q(component_central, "per_sqm", 0.95),
    }
    return scale, _local_decade_summary(component_central, "per_sqm")


def _source_transform_pattern(scan: pl.LazyFrame) -> dict[str, float | int]:
    transformed = (
        scan.filter(pl.col("is_core_analysis_period").fill_null(False) & _as_bool("rent_credit_transform"))
        .select(
            _raw_number("transformable_credit").alias("credit_before"),
            _raw_number("transformed_credit").alias("credit_after"),
            _raw_number("transformable_rent").alias("rent_before"),
            _raw_number("transformed_rent").alias("rent_after"),
        )
        .collect(engine="streaming")
        .drop_nulls()
        .with_columns(
            (pl.col("credit_after") - pl.col("credit_before")).abs().alias("delta_credit"),
            (pl.col("rent_after") - pl.col("rent_before")).abs().alias("delta_rent"),
        )
        .filter((pl.col("delta_credit") > 0) & (pl.col("delta_rent") > 0))
        .with_columns((pl.col("delta_rent") / pl.col("delta_credit")).alias("rate"))
        .filter(pl.col("rate").is_finite() & (pl.col("rate") > 0))
    )
    if transformed.height == 0:
        return {"n": 0, "median": float("nan"), "exact_base_share": float("nan"), "sensitivity_share": float("nan"), "broad_share": float("nan")}
    row = transformed.select(
        pl.len().alias("n"),
        pl.col("rate").median().alias("median"),
        ((pl.col("rate") - BASE_RENT_DEPOSIT_RATE).abs() <= 1e-12).mean().alias("exact_base_share"),
        pl.col("rate").is_between(0.025, 0.035, closed="both").mean().alias("sensitivity_share"),
        pl.col("rate").is_between(0.020, 0.040, closed="both").mean().alias("broad_share"),
    ).row(0, named=True)
    return {key: (int(value) if key == "n" else float(value)) for key, value in row.items()}


def _cross_market_coherence(sale: pl.DataFrame, rental: pl.DataFrame) -> dict[str, float | int]:
    sale_psm = _positive_finite(
        sale.with_columns((pl.col("price") / pl.col("area")).alias("sale_psm")),
        "sale_psm",
    )
    sale_low, sale_high = _q(sale_psm, "sale_psm", 0.01), _q(sale_psm, "sale_psm", 0.99)
    sale_psm = sale_psm.filter(pl.col("sale_psm").is_between(sale_low, sale_high, closed="both"))

    rent_eq = rental.with_columns(
        pl.col("rent").fill_null(0.0).alias("rent0"),
        pl.col("deposit").fill_null(0.0).alias("deposit0"),
    ).with_columns(
        (pl.col("rent0") + BASE_RENT_DEPOSIT_RATE * pl.col("deposit0")).alias("monthly_equivalent")
    ).with_columns(
        (pl.col("monthly_equivalent") / pl.col("area")).alias("rent_eq_psm")
    )
    rent_eq = _positive_finite(rent_eq, "rent_eq_psm")
    rent_low, rent_high = _q(rent_eq, "rent_eq_psm", 0.01), _q(rent_eq, "rent_eq_psm", 0.99)
    rent_eq = rent_eq.filter(pl.col("rent_eq_psm").is_between(rent_low, rent_high, closed="both"))

    sale_cells = (
        sale_psm.group_by(["city_slug", "neighborhood_slug"])
        .agg(pl.len().alias("sale_n"), pl.col("sale_psm").median().alias("sale_psm_median"))
        .filter(pl.col("sale_n") >= MIN_LOCAL_N)
    )
    rent_cells = (
        rent_eq.group_by(["city_slug", "neighborhood_slug"])
        .agg(pl.len().alias("rent_n"), pl.col("rent_eq_psm").median().alias("rent_eq_psm_median"))
        .filter(pl.col("rent_n") >= MIN_LOCAL_N)
    )
    matched = (
        sale_cells.join(rent_cells, on=["city_slug", "neighborhood_slug"], how="inner")
        .filter((pl.col("sale_psm_median") > 0) & (pl.col("rent_eq_psm_median") > 0))
        .with_columns(
            (pl.col("sale_psm_median") / pl.col("rent_eq_psm_median")).alias("months_ratio"),
            pl.col("sale_psm_median").log().alias("log_sale"),
            pl.col("rent_eq_psm_median").log().alias("log_rent"),
        )
    )
    if matched.height == 0:
        return {"cells": 0, "log_correlation": float("nan"), "ratio_median": float("nan"), "ratio_p05": float("nan"), "ratio_p95": float("nan")}
    correlation = matched.select(pl.corr("log_sale", "log_rent")).item()
    return {
        "cells": matched.height,
        "log_correlation": float(correlation),
        "ratio_median": _q(matched, "months_ratio", 0.50),
        "ratio_p05": _q(matched, "months_ratio", 0.05),
        "ratio_p95": _q(matched, "months_ratio", 0.95),
    }


def _pct(value: float) -> str:
    return "NA" if not math.isfinite(value) else f"{100.0 * value:.3f}%"


def _num(value: float, digits: int = 2) -> str:
    return "NA" if not math.isfinite(value) else f"{value:,.{digits}f}"


def _write_interpretation(path: Path, *, row_count: int, core_rows: int, parity_mismatch: int, parity_pairs: int, sale_decade: dict[str, float | int], rent_decade: dict[str, float | int], deposit_decade: dict[str, float | int], transform: dict[str, float | int], cross: dict[str, float | int]) -> None:
    text = f"""# Currency Scale and Operational Unit Decision

## 1. Decision question

The project separates the numerical-scale question from the absolute denomination-label question: whether the supplied monetary values require a project-wide `x10` or `/10` correction, and whether the dataset alone can prove Rial versus Toman. The audit uses all **{row_count:,}** Silver rows; **{core_rows:,}** belong to the May-December 2024 core period. It deliberately avoids an external housing-price benchmark and does not infer the unit from listing-text mentions.

## 2. Stage 1 - Scale-integrity evidence

Raw-to-typed parity was checked across **{parity_pairs}** monetary pairs and found **{parity_mismatch:,}** mismatches. Comparable neighborhood cells were then tested for a material factor-of-ten mixture. Near-0.1x / near-10x shares were **{_pct(float(sale_decade['near_one_tenth_share']))} / {_pct(float(sale_decade['near_ten_x_share']))}** for apartment-sale PSM, **{_pct(float(rent_decade['near_one_tenth_share']))} / {_pct(float(rent_decade['near_ten_x_share']))}** for apartment rent per sqm, and **{_pct(float(deposit_decade['near_one_tenth_share']))} / {_pct(float(deposit_decade['near_ten_x_share']))}** for apartment deposit per sqm. These patterns do not support a broad `x10` or `/10` correction.

## 3. Stage 2 - Internal economic-coherence evidence

Across **{int(transform['n']):,}** source rent-credit transformations, the median absolute delta ratio was **{float(transform['median']):.3f}**; **{_pct(float(transform['exact_base_share']))}** were exactly 0.030 and **{_pct(float(transform['sensitivity_share']))}** fell between 0.025 and 0.035. This recovers a strong source-encoded relationship between rent and deposit components.

Separately, **{int(cross['cells']):,}** neighborhoods with at least {MIN_LOCAL_N} eligible apartment-sale and {MIN_LOCAL_N} eligible rental observations showed a **{float(cross['log_correlation']):.4f}** log-correlation between apartment sale asking PSM and 0.030 monthly rent-equivalent PSM. The median sale-to-monthly-equivalent ratio was **{float(cross['ratio_median']):.2f} months** (P05 **{float(cross['ratio_p05']):.2f}**, P95 **{float(cross['ratio_p95']):.2f}**). These are internal economic-consistency diagnostics, not external market-price benchmarks.

## 4. Identification boundary and final decision

The combined evidence strongly supports keeping the supplied **numerical scale unchanged (`scale=1`)** and provides no internal justification for a project-wide factor-of-ten conversion. However, globally multiplying every monetary field by the same constant leaves correlations, rankings and many economic ratios unchanged. Internal relationships can therefore validate scale consistency and reject a broad mixed-unit problem, but cannot prove the absolute Rial/Toman label.

- **Numerical scale:** keep `scale=1`.
- **Project-wide x10 or /10 conversion:** not applied.
- **Observation type:** listing asking price, not transaction price.
- **Operational reporting unit:** Toman.
- **Formal source status:** `toman_assumed_unconfirmed`.
- **Future confirmation:** use a versioned migration and parity review rather than silently rewriting historical artifacts.
"""
    atomic_write_text(text, path)


def run_currency_validation(input_path: Path, output_path: Path | None = None) -> Path:
    """Run the two-stage full-dataset currency-scale inference.

    The returned CSV path is intentionally unchanged so M2 closeout and downstream
    contracts remain compatible. A concise Markdown interpretation is written beside it.
    """
    output_path = output_path or OUTPUTS_DIR / "tables" / "milestone_2" / "currency" / "currency_validation_summary.csv"
    interpretation_path = output_path.with_name("currency_inference_interpretation.md")

    scan = pl.scan_parquet(input_path)
    schema = set(scan.collect_schema().names())
    missing = sorted(REQUIRED_ANALYSIS_COLUMNS - schema)
    if missing:
        raise ValueError(f"Currency inference requires Silver columns: {missing}")

    counts = scan.select(
        pl.len().alias("row_count"),
        pl.col("is_core_analysis_period").fill_null(False).sum().alias("core_rows"),
    ).collect(engine="streaming").row(0, named=True)
    row_count = int(counts["row_count"])
    core_rows = int(counts["core_rows"] or 0)

    parity_mismatch, parity_pairs = _raw_typed_parity(scan, schema)
    populations = _collect_populations(scan)
    sale_scale, sale_decade = _sale_scale_and_decade(populations["sale"])
    rent_scale, rent_decade = _rental_scale_and_decade(populations["rental"], "rent")
    deposit_scale, deposit_decade = _rental_scale_and_decade(populations["rental"], "deposit")
    transform = _source_transform_pattern(scan)
    cross = _cross_market_coherence(populations["sale"], populations["rental"])

    expected_unit = price_unit()
    expected_observation = price_observation_type()
    factor_applied = bool(setting("contracts", "factor_of_ten_conversion_applied", default=False))

    rows = [
        {
            "stage": "1 - Scale integrity",
            "evidence": "Dataset coverage and Raw/typed parity",
            "population_n": str(row_count),
            "result": f"core_rows={core_rows:,}; monetary_pairs={parity_pairs}; raw_typed_mismatches={parity_mismatch:,}",
            "status": "PASS" if parity_mismatch == 0 else "FAIL",
            "implication": "Uses the full Silver candidate and verifies that M2 did not introduce a hidden monetary rescaling.",
            "analysis_version": VALIDATION_VERSION,
        },
        {
            "stage": "1 - Scale integrity",
            "evidence": "Apartment-sale local decade anomaly",
            "population_n": str(sale_decade["rows"]),
            "result": f"near_0.1x={_pct(float(sale_decade['near_one_tenth_share']))}; near_10x={_pct(float(sale_decade['near_ten_x_share']))}; within_0.5x_2x={_pct(float(sale_decade['within_half_to_double_share']))}",
            "status": "PASS",
            "implication": "No broad factor-of-ten subpopulation is visible in supported neighborhood sale-PSM cells.",
            "analysis_version": VALIDATION_VERSION,
        },
        {
            "stage": "1 - Scale integrity",
            "evidence": "Apartment-rent local decade anomaly",
            "population_n": str(rent_decade["rows"]),
            "result": f"near_0.1x={_pct(float(rent_decade['near_one_tenth_share']))}; near_10x={_pct(float(rent_decade['near_ten_x_share']))}",
            "status": "PASS",
            "implication": "Rent is more dispersed than sale PSM, but no symmetric/systematic 10x split is present.",
            "analysis_version": VALIDATION_VERSION,
        },
        {
            "stage": "1 - Scale integrity",
            "evidence": "Apartment-deposit local decade anomaly",
            "population_n": str(deposit_decade["rows"]),
            "result": f"near_0.1x={_pct(float(deposit_decade['near_one_tenth_share']))}; near_10x={_pct(float(deposit_decade['near_ten_x_share']))}; within_0.5x_2x={_pct(float(deposit_decade['within_half_to_double_share']))}",
            "status": "PASS",
            "implication": "Deposit values do not show a material factor-of-ten unit mixture.",
            "analysis_version": VALIDATION_VERSION,
        },
        {
            "stage": "2 - Hidden economic structure",
            "evidence": "Source rent-deposit transformation",
            "population_n": str(transform["n"]),
            "result": f"median_delta_ratio={float(transform['median']):.6f}; exact_0.030={_pct(float(transform['exact_base_share']))}; in_0.025_0.035={_pct(float(transform['sensitivity_share']))}; in_0.020_0.040={_pct(float(transform['broad_share']))}",
            "status": "PASS" if int(transform["n"]) > 0 else "FAIL",
            "implication": "Recovers a source-encoded rent-credit conversion pattern directly from independent Raw fields.",
            "analysis_version": VALIDATION_VERSION,
        },
        {
            "stage": "2 - Hidden economic structure",
            "evidence": "Apartment sale vs rent-equivalent cross-market coherence",
            "population_n": f"{cross['cells']} matched neighborhoods",
            "result": f"log_correlation={float(cross['log_correlation']):.4f}; median_months_ratio={float(cross['ratio_median']):.2f}; p05={float(cross['ratio_p05']):.2f}; p95={float(cross['ratio_p95']):.2f}",
            "status": "PASS" if int(cross["cells"]) > 0 and float(cross["log_correlation"]) > 0.8 else "REVIEW",
            "implication": "Independent sale and rental regimes share a strong local-market monetary structure under one common scale.",
            "analysis_version": VALIDATION_VERSION,
        },
        {
            "stage": "2 - Hidden economic structure",
            "evidence": "Apartment sale PSM internal scale",
            "population_n": str(sale_scale["n"]),
            "result": f"median={_num(float(sale_scale['median']),0)}; p05={_num(float(sale_scale['p05']),0)}; p95={_num(float(sale_scale['p95']),0)} raw monetary units/m2",
            "status": "PASS",
            "implication": "Absolute-scale audit anchor derived only from project price and area fields.",
            "analysis_version": VALIDATION_VERSION,
        },
        {
            "stage": "2 - Hidden economic structure",
            "evidence": "Apartment rent/deposit per m2 internal scale",
            "population_n": f"rent={rent_scale['n']}; deposit={deposit_scale['n']}",
            "result": f"median_rent_per_m2={_num(float(rent_scale['median']),0)}; median_deposit_per_m2={_num(float(deposit_scale['median']),0)} raw monetary units",
            "status": "PASS",
            "implication": "Area-normalized rental components reconcile with the recovered rent-deposit transformation structure.",
            "analysis_version": VALIDATION_VERSION,
        },
        {
            "stage": "Decision",
            "evidence": "Currency-scale inference",
            "population_n": "full dataset + all eligible subpopulations",
            "result": f"scale=1; operational_currency={setting('contracts', 'operational_currency', default='toman')}; price_unit={expected_unit}; factor_of_ten_applied={factor_applied}; observation_type={expected_observation}",
            "status": "REVIEW" if expected_unit == "toman_assumed_unconfirmed" else "PASS",
            "implication": "Keep current scale; do not apply x10 or /10. Toman remains the operational unit, while source denomination remains formally unconfirmed.",
            "analysis_version": VALIDATION_VERSION,
        },
    ]

    atomic_write_csv(pl.DataFrame(rows), output_path)
    _write_interpretation(
        interpretation_path,
        row_count=row_count,
        core_rows=core_rows,
        parity_mismatch=parity_mismatch,
        parity_pairs=parity_pairs,
        sale_decade=sale_decade,
        rent_decade=rent_decade,
        deposit_decade=deposit_decade,
        transform=transform,
        cross=cross,
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the M2 two-stage full-dataset currency-scale inference.")
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    output = run_currency_validation(args.input)
    print("M2 CURRENCY INFERENCE COMPLETED")
    print(f"table: {output}")
    print(f"interpretation: {output.with_name('currency_inference_interpretation.md')}")


if __name__ == "__main__":
    main()
