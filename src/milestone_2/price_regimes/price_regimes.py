from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from src.common.config import setting
from src.common.io_utils import atomic_write_csv
from src.common.paths import OUTPUTS_DIR

MISSING_TEXT = ["", "null", "<null>", "none", "nan", "n/a", "na"]
PRICE_REGIME_DOMAIN = [
    "sale", "rent_plus_deposit", "full_deposit", "rent_only",
    "rent_negotiable", "rent_unknown_or_incomplete", "temporary_rent",
    "service", "unknown",
]


def _text_present(column: str) -> pl.Expr:
    value = pl.col(column).cast(pl.String).str.strip_chars()
    return (value.is_not_null() & ~value.str.to_lowercase().is_in(MISSING_TEXT)).fill_null(False)


def _price_regime_expr() -> pl.Expr:
    return (
        pl.when(pl.col("market_regime") == "sale").then(pl.lit("sale"))
        .when((pl.col("market_regime") == "long_term_rent") & (pl.col("long_term_rent_regime") == "rent_plus_deposit")).then(pl.lit("rent_plus_deposit"))
        .when((pl.col("market_regime") == "long_term_rent") & (pl.col("long_term_rent_regime") == "full_deposit")).then(pl.lit("full_deposit"))
        .when((pl.col("market_regime") == "long_term_rent") & (pl.col("long_term_rent_regime") == "rent_only")).then(pl.lit("rent_only"))
        .when((pl.col("market_regime") == "long_term_rent") & (pl.col("long_term_rent_regime") == "negotiable")).then(pl.lit("rent_negotiable"))
        .when(pl.col("market_regime") == "long_term_rent").then(pl.lit("rent_unknown_or_incomplete"))
        .when(pl.col("market_regime") == "temporary_rent").then(pl.lit("temporary_rent"))
        .when(pl.col("market_regime") == "service").then(pl.lit("service"))
        .otherwise(pl.lit("unknown"))
    )


def _rule_id_expr() -> pl.Expr:
    return (
        pl.when((pl.col("market_regime") == "sale") & (pl.col("sale_price_status") == "fixed_positive")).then(pl.lit("PR01_SALE_FIXED_POSITIVE"))
        .when((pl.col("market_regime") == "sale") & (pl.col("sale_price_status") == "negotiable")).then(pl.lit("PR02_SALE_NEGOTIABLE"))
        .when((pl.col("market_regime") == "sale") & (pl.col("sale_price_status") == "missing")).then(pl.lit("PR03_SALE_PRICE_MISSING"))
        .when((pl.col("market_regime") == "sale") & (pl.col("sale_price_status") == "semantic_zero_not_sale_price")).then(pl.lit("PR04_SALE_SEMANTIC_ZERO"))
        .when(pl.col("market_regime") == "sale").then(pl.lit("PR05_SALE_PRICE_INCONSISTENT"))
        .when((pl.col("market_regime") == "long_term_rent") & (pl.col("long_term_rent_regime") == "rent_plus_deposit")).then(pl.lit("PR10_RENT_PLUS_DEPOSIT"))
        .when((pl.col("market_regime") == "long_term_rent") & (pl.col("long_term_rent_regime") == "full_deposit")).then(pl.lit("PR11_FULL_DEPOSIT"))
        .when((pl.col("market_regime") == "long_term_rent") & (pl.col("long_term_rent_regime") == "rent_only")).then(pl.lit("PR12_RENT_ONLY"))
        .when((pl.col("market_regime") == "long_term_rent") & (pl.col("long_term_rent_regime") == "negotiable")).then(pl.lit("PR13_RENT_NEGOTIABLE"))
        .when(pl.col("market_regime") == "long_term_rent").then(pl.lit("PR14_RENT_UNKNOWN_OR_INCOMPLETE"))
        .when(pl.col("market_regime") == "temporary_rent").then(pl.lit("PR20_TEMPORARY_RENT"))
        .when(pl.col("market_regime") == "service").then(pl.lit("PR30_REAL_ESTATE_SERVICE"))
        .otherwise(pl.lit("PR99_UNKNOWN_CATEGORY"))
    )


def apply_price_regimes(frame: pl.LazyFrame) -> pl.LazyFrame:
    """Classify price regimes without converting the project's assumed currency unit."""
    version = str(setting("milestone_2", "versions", "price_regime", default="price-regime-v1"))
    cleaned = frame.with_columns(
        [
            pl.when(pl.col("rent_component_status") == "fixed_positive").then(pl.col("rent_value_toman"))
            .when(pl.col("rent_component_status") == "semantic_zero").then(pl.lit(0, dtype=pl.Int64))
            .otherwise(pl.lit(None, dtype=pl.Int64)).alias("monthly_rent_clean_toman"),
            pl.when(pl.col("credit_component_status") == "fixed_positive").then(pl.col("credit_value_toman"))
            .when(pl.col("credit_component_status") == "semantic_zero").then(pl.lit(0, dtype=pl.Int64))
            .otherwise(pl.lit(None, dtype=pl.Int64)).alias("deposit_clean_toman"),
        ]
    )
    price_present = _text_present("price_mode") | pl.col("price_value_toman").is_not_null()
    rent_present = _text_present("rent_mode") | pl.col("rent_value_toman").is_not_null()
    credit_present = _text_present("credit_mode") | pl.col("credit_value_toman").is_not_null()
    any_component = price_present | rent_present | credit_present
    conflict = ((pl.col("is_sale") & (rent_present | credit_present)) | (pl.col("is_long_term_rent") & price_present)).fill_null(False)

    classified = cleaned.with_columns(
        [
            _price_regime_expr().alias("price_regime"),
            _rule_id_expr().alias("price_regime_rule_id"),
            conflict.alias("price_regime_conflict_flag"),
            pl.lit(version).alias("price_regime_version"),
        ]
    )
    return classified.with_columns(
        [
            (pl.col("price_regime") == "unknown").fill_null(True).alias("price_regime_unclassified_flag"),
            (
                pl.col("price_regime_conflict_flag")
                | pl.col("price_regime").is_in(["rent_negotiable", "rent_unknown_or_incomplete", "unknown"])
                | ((pl.col("market_regime") == "sale") & (pl.col("sale_price_status") != "fixed_positive"))
                | (pl.col("market_regime").is_in(["temporary_rent", "service"]) & any_component)
            ).fill_null(True).alias("price_regime_review_flag"),
        ]
    )


def write_price_regime_review(input_path: Path, output_path: Path | None = None) -> Path:
    """Write the compact §39 review table from a Silver candidate/master."""
    output_path = output_path or OUTPUTS_DIR / "tables" / "milestone_2" / "price_regimes" / "price_regime_review_summary.csv"
    frame = pl.scan_parquet(input_path)
    patterns = [
        ("sale + fixed positive asking price", "sale", pl.col("price_regime") == "sale", "Keep fixed positive sale asking price; non-fixed sale values stay non-analytical."),
        ("long-term rent + positive rent + positive deposit", "rent_plus_deposit", pl.col("price_regime") == "rent_plus_deposit", "Preserve both fixed components."),
        ("long-term rent + zero rent + positive deposit", "full_deposit", pl.col("price_regime") == "full_deposit", "Preserve semantic zero rent and positive deposit."),
        ("long-term rent + positive rent + zero deposit", "rent_only", pl.col("price_regime") == "rent_only", "Preserve positive rent and semantic zero deposit."),
        ("long-term rent + negotiable component", "rent_negotiable", pl.col("price_regime") == "rent_negotiable", "Do not invent numeric values; retain for review."),
        ("long-term rent + missing/incomplete components", "rent_unknown_or_incomplete", pl.col("price_regime") == "rent_unknown_or_incomplete", "Do not force incomplete records into a numeric rent regime."),
        ("temporary rental category", "temporary_rent", pl.col("price_regime") == "temporary_rent", "Keep separate from long-term rent."),
        ("real-estate service category", "service", pl.col("price_regime") == "service", "Keep outside property price analysis."),
        ("unrecognized category", "unknown", pl.col("price_regime") == "unknown", "Retain and review; do not guess a regime."),
    ]
    exprs = [condition.sum().alias(f"n_{i}") for i, (_, _, condition, _) in enumerate(patterns)]
    counts = frame.select(exprs).collect(engine="streaming").row(0, named=True)
    rows = []
    for i, (raw_pattern, expected, _, rule) in enumerate(patterns):
        count = int(counts[f"n_{i}"] or 0)
        result = "REVIEW" if expected in {"rent_negotiable", "rent_unknown_or_incomplete", "unknown"} and count else "PASS"
        rows.append({
            "Raw Pattern": raw_pattern,
            "Expected Regime": expected,
            "Observed Count": count,
            "Cleaning Rule": rule,
            "Validation Result": result,
        })
    atomic_write_csv(pl.DataFrame(rows), output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the canonical M2 price-regime review table.")
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    path = write_price_regime_review(args.input)
    print(path)


if __name__ == "__main__":
    main()
