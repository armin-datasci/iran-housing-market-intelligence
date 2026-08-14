from __future__ import annotations

import polars as pl

from src.common.config import setting
from src.common.domain import property_family_categories


def _build_bounds(
    frame: pl.LazyFrame,
    value_column: str,
    group_columns: list[str],
    min_rows: int,
    multiplier: float,
    label: str,
) -> pl.LazyFrame:
    n = f"_{label}_n"
    q1 = f"_{label}_q1"
    q3 = f"_{label}_q3"
    iqr = f"_{label}_iqr"
    lower = f"_{label}_lower"
    upper = f"_{label}_upper"
    return (
        frame.filter(
            pl.col("is_core_analysis_period")
            & pl.col("supply_keep_flag")
            & pl.col(value_column).is_not_null()
            & (pl.col(value_column) > 0)
        )
        .group_by(group_columns)
        .agg([
            pl.len().alias(n),
            pl.col(value_column).quantile(0.25).alias(q1),
            pl.col(value_column).quantile(0.75).alias(q3),
        ])
        .with_columns((pl.col(q3) - pl.col(q1)).alias(iqr))
        .filter((pl.col(n) >= min_rows) & (pl.col(iqr) > 0))
        .with_columns([
            pl.max_horizontal(pl.lit(0.0), pl.col(q1) - multiplier * pl.col(iqr)).alias(lower),
            (pl.col(q3) + multiplier * pl.col(iqr)).alias(upper),
        ])
        .select(group_columns + [n, lower, upper])
    )


def _attach_flag(
    frame: pl.LazyFrame,
    reference: pl.LazyFrame,
    *,
    value_column: str,
    local_keys: list[str],
    fallback_keys: list[str],
    prefix: str,
    multiplier: float,
    min_local: int,
    min_fallback: int,
) -> pl.LazyFrame:
    local_label = f"{prefix}_local"
    fallback_label = f"{prefix}_fallback"
    local = _build_bounds(reference, value_column, local_keys, min_local, multiplier, local_label)
    fallback = _build_bounds(reference, value_column, fallback_keys, min_fallback, multiplier, fallback_label)
    joined = frame.join(local, on=local_keys, how="left").join(fallback, on=fallback_keys, how="left")
    lower = pl.coalesce([pl.col(f"_{local_label}_lower"), pl.col(f"_{fallback_label}_lower")])
    upper = pl.coalesce([pl.col(f"_{local_label}_upper"), pl.col(f"_{fallback_label}_upper")])
    flag = (
        pl.col(value_column).is_not_null()
        & (pl.col(value_column) > 0)
        & lower.is_not_null()
        & ((pl.col(value_column) < lower) | (pl.col(value_column) > upper))
    ).fill_null(False)
    return joined.with_columns(flag.alias(f"{prefix}_outlier_flag")).drop([
        f"_{local_label}_n", f"_{local_label}_lower", f"_{local_label}_upper",
        f"_{fallback_label}_n", f"_{fallback_label}_lower", f"_{fallback_label}_upper",
    ])


def apply_outlier_rules(frame: pl.LazyFrame) -> pl.LazyFrame:
    """Apply flag-only business and grouped-IQR outlier policy; rows are never deleted."""
    multiplier = float(setting("milestone_2", "outliers", "canonical_iqr_multiplier", default=3.0))
    min_local = int(setting("milestone_2", "outliers", "min_local_group_rows", default=50))
    min_fallback = int(setting("milestone_2", "outliers", "min_fallback_group_rows", default=200))
    version = str(setting("milestone_2", "versions", "outlier", default="outlier-policy-m2-v2"))
    families = property_family_categories()
    standard_building = sorted(families.get("apartment", set()) | families.get("commercial", set()) | families.get("villa", set()))
    villa = sorted(families.get("villa", set()))
    land = sorted(families.get("land", set()))
    industrial = sorted(families.get("industrial", set()))

    frame = frame.with_columns([
        pl.when(pl.col("cat3_slug").is_in(standard_building)).then(pl.col("building_size_sqm")).otherwise(None).alias("effective_building_area_sqm"),
        pl.when(pl.col("cat3_slug").is_in(land)).then(pl.col("building_size_sqm"))
        .when(pl.col("cat3_slug").is_in(villa)).then(pl.col("land_size_sqm"))
        .otherwise(None).alias("effective_land_area_sqm"),
        pl.when(pl.col("cat3_slug").is_in(land)).then(pl.lit("building_size_sqm"))
        .when(pl.col("cat3_slug").is_in(villa)).then(pl.lit("land_size_sqm"))
        .otherwise(None).alias("effective_land_area_source"),
        pl.col("cat3_slug").is_in(sorted((families.get("apartment", set()) | families.get("commercial", set()) | families.get("land", set())))).fill_null(False).alias("primary_area_required_flag"),
        pl.col("cat3_slug").is_in(industrial).fill_null(False).alias("area_manual_review_flag"),
    ])
    frame = frame.with_columns([
        (pl.col("primary_area_required_flag") & pl.col("primary_area_sqm").is_null()).alias("primary_area_missing_flag"),
        (pl.col("primary_area_required_flag") & pl.col("primary_area_sqm").is_not_null() & (pl.col("primary_area_sqm") <= 0)).alias("primary_area_nonpositive_flag"),
    ])

    reference = frame
    specs = [
        ("primary_area", "primary_area_sqm", ["city_slug", "cat3_slug"], ["cat3_slug"]),
        ("building_area", "effective_building_area_sqm", ["city_slug", "cat3_slug"], ["cat3_slug"]),
        ("land_area", "effective_land_area_sqm", ["city_slug", "cat3_slug"], ["cat3_slug"]),
        ("sale_price_per_sqm", "price_per_sqm_raw_toman", ["city_slug", "cat3_slug"], ["cat3_slug"]),
        ("monthly_rent", "monthly_rent_clean_toman", ["city_slug", "cat3_slug", "price_regime"], ["cat3_slug", "price_regime"]),
        ("deposit", "deposit_clean_toman", ["city_slug", "cat3_slug", "price_regime"], ["cat3_slug", "price_regime"]),
    ]
    for prefix, value_column, local_keys, fallback_keys in specs:
        frame = _attach_flag(
            frame,
            reference,
            value_column=value_column,
            local_keys=local_keys,
            fallback_keys=fallback_keys,
            prefix=prefix,
            multiplier=multiplier,
            min_local=min_local,
            min_fallback=min_fallback,
        )

    frame = frame.with_columns([
        (
            pl.col("primary_area_nonpositive_flag")
            | pl.col("primary_area_outlier_flag")
            | pl.col("building_area_outlier_flag")
            | pl.col("land_area_outlier_flag")
        ).fill_null(False).alias("outlier_area_flag"),
        (
            pl.col("sale_price_per_sqm_outlier_flag")
            | pl.col("monthly_rent_outlier_flag")
            | pl.col("deposit_outlier_flag")
        ).fill_null(False).alias("outlier_price_flag"),
        (
            pl.col("construction_year_before_1370_flag")
            | pl.col("construction_year_numeric_invalid_flag")
        ).fill_null(False).alias("outlier_year_flag"),
        pl.lit(version).alias("outlier_rule_version"),
    ])
    return frame
