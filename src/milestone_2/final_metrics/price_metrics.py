from __future__ import annotations

import polars as pl

from src.common.config import price_observation_type, price_unit, setting

PROPERTY_LISTING_REGIMES = [
    "sale", "rent_plus_deposit", "full_deposit", "rent_only",
    "rent_negotiable", "rent_unknown_or_incomplete", "temporary_rent",
]


def _monthly_equivalent(k_toman: int) -> pl.Expr:
    factor = float(k_toman) / 1_000_000.0
    return (
        pl.col("monthly_rent_clean_toman").cast(pl.Float64)
        + pl.col("deposit_clean_toman").cast(pl.Float64) * factor
    ).round(0).cast(pl.Int64)


def _deposit_equivalent(k_toman: int) -> pl.Expr:
    return (
        pl.col("deposit_clean_toman").cast(pl.Float64)
        + pl.col("monthly_rent_clean_toman").cast(pl.Float64) / float(k_toman) * 1_000_000.0
    ).round(0).cast(pl.Int64)


def apply_final_metrics(frame: pl.LazyFrame) -> pl.LazyFrame:
    """Create canonical asking-price metrics and final analytical eligibility flags."""
    low = int(setting("analysis", "rent_equivalence_k_toman", "low", default=25_000))
    base = int(setting("analysis", "rent_equivalence_k_toman", "base", default=30_000))
    high = int(setting("analysis", "rent_equivalence_k_toman", "high", default=35_000))
    rent_version = str(setting("milestone_2", "versions", "rent_equivalent", default="rent-equivalent-sensitivity-v1"))
    quality_version = str(setting("milestone_2", "versions", "quality", default="quality-gate-m2-v2"))
    silver_version = str(setting("milestone_2", "versions", "silver_master", default="silver-master-v1"))

    frame = frame.with_columns([
        (
            pl.col("is_core_analysis_period")
            & (pl.col("price_regime") == "sale")
            & pl.col("sale_price_clean_toman").is_not_null()
            & (pl.col("sale_price_clean_toman") > 0)
            & (pl.col("primary_area_sqm") > 0)
            & ~pl.col("outlier_area_flag")
            & ~pl.col("sale_price_per_sqm_outlier_flag")
            & ~pl.col("price_regime_conflict_flag")
            & (pl.col("type_parse_error_count") == 0)
        ).fill_null(False).alias("sale_price_analysis_eligible_flag"),
        (
            pl.col("is_core_analysis_period")
            & pl.col("price_regime").is_in(["rent_plus_deposit", "full_deposit", "rent_only"])
            & ~pl.col("monthly_rent_outlier_flag")
            & ~pl.col("deposit_outlier_flag")
            & ~pl.col("price_regime_conflict_flag")
            & (pl.col("type_parse_error_count") == 0)
        ).fill_null(False).alias("rent_component_analysis_eligible_flag"),
    ])

    frame = frame.with_columns([
        (
            pl.col("is_core_analysis_period")
            & (pl.col("price_regime") == "sale")
            & pl.col("sale_price_clean_toman").is_not_null()
            & (pl.col("sale_price_clean_toman") > 0)
            & ~pl.col("price_regime_conflict_flag")
            & (pl.col("type_parse_error_count") == 0)
            & pl.col("supply_keep_flag")
        ).fill_null(False).alias("sale_total_price_analysis_eligible_flag"),
        (pl.col("sale_price_analysis_eligible_flag") & pl.col("supply_keep_flag")).fill_null(False).alias("sale_price_per_sqm_final_eligible_flag"),
        (pl.col("rent_component_analysis_eligible_flag") & pl.col("supply_keep_flag")).fill_null(False).alias("rent_final_eligible_flag"),
        (
            pl.col("is_core_analysis_period")
            & pl.col("price_regime").is_in(PROPERTY_LISTING_REGIMES)
            & pl.col("supply_keep_flag")
            & (pl.col("type_parse_error_count") == 0)
        ).fill_null(False).alias("supply_analysis_eligible_flag"),
        (
            pl.col("coordinate_pair_present")
            & pl.col("geo_country_valid")
            & ~pl.col("coordinate_partial_flag")
            & (pl.col("type_parse_error_count") == 0)
        ).fill_null(False).alias("geo_aggregate_map_eligible_flag"),
    ])

    frame = frame.with_columns([
        (pl.col("supply_analysis_eligible_flag") & pl.col("geo_aggregate_map_eligible_flag")).fill_null(False).alias("map_analysis_eligible_flag"),
        pl.when(pl.col("sale_total_price_analysis_eligible_flag")).then(pl.col("sale_price_clean_toman")).otherwise(None).alias("sale_price_final_toman"),
        pl.when(pl.col("sale_price_per_sqm_final_eligible_flag")).then(pl.col("price_per_sqm_raw_toman")).otherwise(None).alias("sale_price_per_sqm_final_toman"),
        pl.when(pl.col("rent_final_eligible_flag")).then(pl.col("monthly_rent_clean_toman")).otherwise(None).alias("monthly_rent_final_toman"),
        pl.when(pl.col("rent_final_eligible_flag")).then(pl.col("deposit_clean_toman")).otherwise(None).alias("deposit_final_toman"),
        pl.when(pl.col("rent_final_eligible_flag")).then(_monthly_equivalent(low)).otherwise(None).alias("rent_equivalent_monthly_low_toman"),
        pl.when(pl.col("rent_final_eligible_flag")).then(_monthly_equivalent(base)).otherwise(None).alias("rent_equivalent_monthly_base_toman"),
        pl.when(pl.col("rent_final_eligible_flag")).then(_monthly_equivalent(high)).otherwise(None).alias("rent_equivalent_monthly_high_toman"),
        pl.when(pl.col("rent_final_eligible_flag")).then(_deposit_equivalent(low)).otherwise(None).alias("rent_equivalent_deposit_low_toman"),
        pl.when(pl.col("rent_final_eligible_flag")).then(_deposit_equivalent(base)).otherwise(None).alias("rent_equivalent_deposit_base_toman"),
        pl.when(pl.col("rent_final_eligible_flag")).then(_deposit_equivalent(high)).otherwise(None).alias("rent_equivalent_deposit_high_toman"),
        pl.lit(rent_version).alias("rent_equivalent_rule_version"),
        pl.lit(price_unit()).alias("price_unit"),
        pl.lit(price_observation_type()).alias("price_observation_type"),
        pl.lit(silver_version).alias("silver_master_version"),
    ])

    issue_exprs = [
        pl.col("type_parse_error_count") > 0,
        pl.col("price_regime_conflict_flag"),
        pl.col("price_regime_review_flag"),
        pl.col("primary_area_missing_flag"),
        pl.col("outlier_area_flag"),
        pl.col("outlier_price_flag"),
        pl.col("outlier_year_flag"),
        pl.col("exact_duplicate_flag"),
        pl.col("same_month_duplicate_flag"),
        pl.col("coordinate_partial_flag"),
        ~pl.col("geo_country_valid") & pl.col("coordinate_pair_present"),
        pl.col("area_manual_review_flag"),
    ]
    return frame.with_columns([
        pl.sum_horizontal([expr.fill_null(False).cast(pl.UInt8) for expr in issue_exprs]).cast(pl.UInt16).alias("quality_issue_count"),
        pl.lit(quality_version).alias("quality_rule_version"),
    ]).with_columns((pl.col("quality_issue_count") > 0).alias("record_quality_review_flag"))
