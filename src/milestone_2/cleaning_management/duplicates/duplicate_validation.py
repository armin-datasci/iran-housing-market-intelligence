from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from src.common.config import setting
from src.common.io_utils import atomic_write_csv
from src.common.paths import OUTPUTS_DIR

ADVANCED_ER_METHOD = "deterministic_multi_pass_record_linkage"
ADVANCED_ER_CLUSTER_COVERAGE_MIN = 1.0

RAW_COLUMNS = [
    "cat2_slug", "cat3_slug", "city_slug", "neighborhood_slug", "created_at_month", "user_type",
    "description", "title", "rent_mode", "rent_value", "rent_to_single", "rent_type",
    "price_mode", "price_value", "credit_mode", "credit_value", "rent_credit_transform",
    "transformable_price", "transformable_credit", "transformed_credit", "transformable_rent",
    "transformed_rent", "land_size", "building_size", "deed_type", "has_business_deed",
    "floor", "rooms_count", "total_floors_count", "unit_per_floor", "has_balcony",
    "has_elevator", "has_warehouse", "has_parking", "construction_year", "is_rebuilt",
    "has_water", "has_warm_water_provider", "has_electricity", "has_gas", "has_heating_system",
    "has_cooling_system", "has_restroom", "has_security_guard", "has_barbecue", "building_direction",
    "has_pool", "has_jacuzzi", "has_sauna", "floor_material", "property_type",
    "regular_person_capacity", "extra_person_capacity", "cost_per_extra_person",
    "rent_price_on_regular_days", "rent_price_on_special_days", "rent_price_at_weekends",
    "location_latitude", "location_longitude", "location_radius",
]


def _clean_text(column: str) -> pl.Expr:
    return (
        pl.col(column).cast(pl.String).fill_null("").str.to_lowercase()
        .str.replace_all("ي", "ی", literal=True).str.replace_all("ك", "ک", literal=True)
        .str.replace_all(r"[^\p{L}\p{N}]+", " ").str.replace_all(r"\s+", " ").str.strip_chars()
    )


def add_duplicate_features(frame: pl.LazyFrame) -> pl.LazyFrame:
    """Scalable deterministic ER: exact, high-confidence text+structure, and medium blocking.

    Only exact/high-confidence same-month excess rows are removed from supply counts.
    Medium candidates and cross-month repeats are retained and only flagged for review.
    """
    version = str(setting("milestone_2", "versions", "deduplication", default="deduplication-v2"))
    area_bin = float(setting("milestone_2", "duplicates", "medium_area_bin_sqm", default=5.0))
    price_bin = int(setting("milestone_2", "duplicates", "medium_price_bin_toman", default=10_000_000))

    base = frame.with_columns(
        [
            pl.col("analysis_month").cast(pl.String).str.slice(0, 7).alias("_duplicate_month"),
            _clean_text("title_normalized").alias("_dup_title"),
            _clean_text("description_normalized").alias("_dup_description"),
        ]
    )

    raw_struct = pl.struct([pl.col(c) for c in RAW_COLUMNS])
    high_fields = [
        "_dup_title", "_dup_description", "cat2_slug", "cat3_slug", "city_slug",
        "neighborhood_slug", "user_type", "price_value", "rent_value", "credit_value",
        "building_size", "land_size", "floor", "rooms_count", "construction_year",
        "has_parking", "has_elevator", "has_warehouse", "location_latitude", "location_longitude",
    ]
    high_ok = (
        (pl.col("_dup_title").str.len_chars() >= 5)
        & (pl.col("_dup_description").str.len_chars() >= 20)
        & pl.col("city_slug").is_not_null()
        & pl.col("cat3_slug").is_not_null()
    )

    base = base.with_columns(
        [
            raw_struct.hash(seed=401, seed_1=409, seed_2=419, seed_3=421).alias("_exact_h1"),
            raw_struct.hash(seed=503, seed_1=509, seed_2=521, seed_3=523).alias("_exact_h2"),
            pl.when(high_ok).then(pl.struct(high_fields).hash(seed=211, seed_1=223, seed_2=227, seed_3=229)).otherwise(None).alias("_high_h1"),
            pl.when(high_ok).then(pl.struct(high_fields).hash(seed=307, seed_1=311, seed_2=313, seed_3=317)).otherwise(None).alias("_high_h2"),
            ((pl.col("building_size_sqm") / area_bin).round(0).cast(pl.Int64)).alias("_area_bin"),
            ((pl.col("price_value_toman") / price_bin).round(0).cast(pl.Int64)).alias("_price_bin"),
        ]
    )

    medium_ok = (
        pl.col("city_slug").is_not_null() & pl.col("neighborhood_slug").is_not_null()
        & pl.col("cat3_slug").is_not_null() & pl.col("_duplicate_month").is_not_null()
        & pl.col("_dup_title").is_not_null() & (pl.col("_dup_title").str.len_chars() >= 5)
        & pl.col("_area_bin").is_not_null() & pl.col("rooms_count_num").is_not_null()
        & pl.col("_price_bin").is_not_null()
    )
    medium_fields = [
        "_duplicate_month", "city_slug", "neighborhood_slug", "cat3_slug", "user_type",
        "_dup_title", "_area_bin", "rooms_count_num", "_price_bin",
    ]
    base = base.with_columns(
        [
            pl.when(medium_ok).then(pl.struct(medium_fields).hash(seed=601, seed_1=607, seed_2=613, seed_3=617)).otherwise(None).alias("_medium_h1"),
            pl.when(medium_ok).then(pl.struct(medium_fields).hash(seed=701, seed_1=709, seed_2=719, seed_3=727)).otherwise(None).alias("_medium_h2"),
        ]
    )

    exact_groups = (
        base.group_by(["_exact_h1", "_exact_h2"]).agg(
            [pl.len().alias("_exact_size"), pl.col("source_row_id").min().alias("_exact_keep")]
        ).filter(pl.col("_exact_size") > 1)
    )
    high_groups = (
        base.filter(pl.col("_high_h1").is_not_null())
        .group_by(["_duplicate_month", "_high_h1", "_high_h2"])
        .agg([pl.len().alias("_high_size"), pl.col("source_row_id").min().alias("_high_keep")])
        .filter(pl.col("_high_size") > 1)
    )
    high_cross_month = (
        base.filter(pl.col("_high_h1").is_not_null())
        .group_by(["_high_h1", "_high_h2"])
        .agg([
            pl.len().alias("cross_month_repeat_row_count"),
            pl.col("_duplicate_month").n_unique().alias("cross_month_repeat_month_count"),
            pl.col("_duplicate_month").min().alias("cross_month_first_month"),
            pl.col("_duplicate_month").max().alias("cross_month_last_month"),
        ])
        .filter(pl.col("cross_month_repeat_month_count") > 1)
    )
    medium_groups = (
        base.filter(pl.col("_medium_h1").is_not_null())
        .group_by(["_medium_h1", "_medium_h2"])
        .agg(pl.len().alias("_medium_size"))
        .filter(pl.col("_medium_size") > 1)
    )

    joined = (
        base.join(exact_groups, on=["_exact_h1", "_exact_h2"], how="left")
        .join(high_groups, on=["_duplicate_month", "_high_h1", "_high_h2"], how="left")
        .join(high_cross_month, on=["_high_h1", "_high_h2"], how="left")
        .join(medium_groups, on=["_medium_h1", "_medium_h2"], how="left")
        .with_columns([
            pl.col("_exact_size").fill_null(1).cast(pl.UInt32),
            pl.col("_high_size").fill_null(1).cast(pl.UInt32),
            pl.col("_medium_size").fill_null(1).cast(pl.UInt32),
            pl.col("cross_month_repeat_row_count").fill_null(1).cast(pl.UInt32),
            pl.col("cross_month_repeat_month_count").fill_null(1).cast(pl.UInt16),
        ])
    )

    exact_flag = pl.col("_exact_size") > 1
    exact_excess = exact_flag & (pl.col("source_row_id") != pl.col("_exact_keep"))
    high_flag = pl.col("_high_size") > 1
    high_excess = high_flag & (pl.col("source_row_id") != pl.col("_high_keep"))
    medium_flag = (pl.col("_medium_size") > 1) & ~high_flag & ~exact_flag
    cross_flag = pl.col("cross_month_repeat_month_count") > 1

    joined = joined.with_columns(
        [
            exact_flag.fill_null(False).alias("exact_duplicate_flag"),
            exact_excess.fill_null(False).alias("exact_duplicate_excess_flag"),
            high_flag.fill_null(False).alias("same_month_duplicate_flag"),
            high_excess.fill_null(False).alias("same_month_duplicate_excess_flag"),
            pl.col("_high_size").alias("same_month_duplicate_group_size"),
            cross_flag.fill_null(False).alias("cross_month_repeat_flag"),
            pl.when(exact_flag).then(pl.lit("exact"))
            .when(high_flag).then(pl.lit("high"))
            .when(medium_flag).then(pl.lit("medium"))
            .otherwise(pl.lit("none")).alias("deduplication_confidence"),
            pl.when(high_flag | cross_flag).then(pl.concat_str([pl.lit("PDH"), pl.col("_high_h1").cast(pl.String), pl.col("_high_h2").cast(pl.String)], separator="-"))
            .when(medium_flag).then(pl.concat_str([pl.lit("PDM"), pl.col("_medium_h1").cast(pl.String), pl.col("_medium_h2").cast(pl.String)], separator="-"))
            .otherwise(pl.lit(None, dtype=pl.String)).alias("probable_duplicate_cluster_id"),
            (~(exact_excess | high_excess)).fill_null(True).alias("supply_keep_flag"),
            pl.lit(version).alias("deduplication_rule_version"),
        ]
    )

    return joined.drop([
        "_duplicate_month", "_dup_title", "_dup_description", "_exact_h1", "_exact_h2",
        "_exact_size", "_exact_keep", "_high_h1", "_high_h2", "_high_size", "_high_keep",
        "_medium_h1", "_medium_h2", "_medium_size", "_area_bin", "_price_bin",
    ])


def write_duplicate_reports(
    input_path: Path,
    output_dir: Path | None = None,
    qa_dir: Path | None = None,
) -> tuple[Path, Path, Path]:
    output_dir = output_dir or OUTPUTS_DIR / "tables" / "milestone_2" / "duplicates"
    qa_dir = qa_dir or OUTPUTS_DIR / "qa" / "milestone_2" / "duplicates"
    scan = pl.scan_parquet(input_path)
    er_candidate = (
        pl.col("deduplication_confidence").is_in(["high", "medium"])
        | pl.col("cross_month_repeat_flag").cast(pl.Boolean, strict=False).fill_null(False)
    )
    stats = scan.select([
        pl.len().alias("row_count"),
        pl.col("exact_duplicate_flag").sum().alias("exact_duplicate_rows"),
        pl.col("exact_duplicate_excess_flag").sum().alias("exact_duplicate_excess_rows"),
        pl.col("same_month_duplicate_flag").sum().alias("high_probable_duplicate_rows"),
        pl.col("same_month_duplicate_excess_flag").sum().alias("high_probable_duplicate_excess_rows"),
        (pl.col("deduplication_confidence") == "medium").sum().alias("medium_candidate_rows"),
        pl.col("cross_month_repeat_flag").sum().alias("cross_month_repeat_rows"),
        pl.col("supply_keep_flag").sum().alias("conservative_supply_rows"),
        er_candidate.sum().alias("entity_resolution_candidate_rows"),
        (er_candidate & pl.col("probable_duplicate_cluster_id").is_not_null()).sum().alias("entity_resolution_clustered_candidate_rows"),
        pl.col("probable_duplicate_cluster_id").filter(er_candidate).drop_nulls().n_unique().alias("entity_resolution_cluster_count"),
    ]).collect(engine="streaming").row(0, named=True)

    er_candidate_rows = int(stats["entity_resolution_candidate_rows"] or 0)
    er_clustered_rows = int(stats["entity_resolution_clustered_candidate_rows"] or 0)
    er_cluster_count = int(stats["entity_resolution_cluster_count"] or 0)
    er_cluster_coverage = (er_clustered_rows / er_candidate_rows) if er_candidate_rows else 0.0
    cluster_diagnostics = (
        scan.filter(er_candidate & pl.col("probable_duplicate_cluster_id").is_not_null())
        .with_columns(pl.col("analysis_month").cast(pl.String).str.slice(0, 7).alias("_er_month"))
        .group_by("probable_duplicate_cluster_id")
        .agg([
            pl.len().alias("cluster_size"),
            pl.col("_er_month").n_unique().alias("month_count"),
        ])
        .select([
            pl.col("cluster_size").max().fill_null(0).alias("entity_resolution_max_cluster_size"),
            (pl.col("month_count") > 1).sum().alias("entity_resolution_multi_month_cluster_count"),
        ])
        .collect(engine="streaming")
        .row(0, named=True)
    )
    er_bonus_ready = bool(
        er_candidate_rows > 0
        and er_cluster_count > 0
        and er_cluster_coverage >= ADVANCED_ER_CLUSTER_COVERAGE_MIN
    )

    summary_values = dict(stats)
    summary_values.update({
        "entity_resolution_method": ADVANCED_ER_METHOD,
        "entity_resolution_cluster_coverage_rate": f"{er_cluster_coverage:.6f}",
        "entity_resolution_max_cluster_size": int(cluster_diagnostics["entity_resolution_max_cluster_size"] or 0),
        "entity_resolution_multi_month_cluster_count": int(cluster_diagnostics["entity_resolution_multi_month_cluster_count"] or 0),
        "advanced_entity_resolution_bonus_ready": str(er_bonus_ready).lower(),
        "entity_resolution_automatic_exclusion_policy": "exact_or_high_same_month_excess_only",
        "entity_resolution_cross_month_policy": "retain_and_flag",
    })
    summary = pl.DataFrame({"metric": list(summary_values.keys()), "value": [str(v) for v in summary_values.values()]})
    summary_path = output_dir / "duplicate_summary.csv"
    atomic_write_csv(summary, summary_path)

    supply = (
        scan.with_columns(pl.col("analysis_month").cast(pl.String).str.slice(0, 7).alias("month"))
        .group_by("month").agg([
            pl.len().alias("raw_listing_count"),
            pl.col("supply_keep_flag").sum().alias("deduplicated_listing_count"),
            pl.col("exact_duplicate_excess_flag").sum().alias("exact_excess_count"),
            pl.col("same_month_duplicate_excess_flag").sum().alias("high_probable_excess_count"),
            pl.col("cross_month_repeat_flag").sum().alias("cross_month_repeat_rows"),
        ])
        .with_columns(((pl.col("raw_listing_count") - pl.col("deduplicated_listing_count")) / pl.col("raw_listing_count")).alias("duplicate_impact_rate"))
        .sort("month").collect(engine="streaming")
    )
    supply_path = output_dir / "duplicate_supply_impact.csv"
    atomic_write_csv(supply, supply_path)

    review = (
        scan.filter(
            pl.col("deduplication_confidence").is_in(["exact", "high", "medium"])
            | pl.col("cross_month_repeat_flag").cast(pl.Boolean, strict=False).fill_null(False)
        )
        .with_columns([
            (
                pl.col("deduplication_confidence").is_in(["high", "medium"])
                | pl.col("cross_month_repeat_flag").cast(pl.Boolean, strict=False).fill_null(False)
            ).alias("entity_resolution_candidate_flag"),
            pl.when(pl.col("deduplication_confidence") == "exact").then(pl.lit("exact_duplicate"))
            .when(pl.col("deduplication_confidence") == "high").then(pl.lit("high_confidence_same_month_link"))
            .when(pl.col("deduplication_confidence") == "medium").then(pl.lit("medium_confidence_blocked_candidate"))
            .when(pl.col("cross_month_repeat_flag").cast(pl.Boolean, strict=False).fill_null(False)).then(pl.lit("cross_month_link_retained"))
            .otherwise(pl.lit("review")).alias("entity_resolution_role"),
        ])
        .select([
            "source_row_id", "analysis_month", "city_slug", "neighborhood_slug", "cat3_slug",
            "deduplication_confidence", "probable_duplicate_cluster_id", "entity_resolution_candidate_flag",
            "entity_resolution_role", "exact_duplicate_flag", "same_month_duplicate_flag",
            "cross_month_repeat_flag", "supply_keep_flag", "title",
        ])
        .sort(["deduplication_confidence", "probable_duplicate_cluster_id", "analysis_month", "source_row_id"])
        .limit(300).collect(engine="streaming")
    )
    review_path = qa_dir / "duplicate_review_sample.csv"
    atomic_write_csv(review, review_path)
    return summary_path, supply_path, review_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Write M2 duplicate/ER reports from a Silver candidate/master.")
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    for path in write_duplicate_reports(args.input):
        print(path)


if __name__ == "__main__":
    main()
