from __future__ import annotations

from pathlib import Path

import polars as pl

from src.common.config import setting
from src.milestone_2.cleaning_management.duplicates.duplicate_validation import (
    RAW_COLUMNS,
    add_duplicate_features,
    write_duplicate_reports,
)
from src.milestone_2.final_metrics.price_metrics import apply_final_metrics
from src.milestone_2.milestone2_closeout import REQUIRED_OUTPUTS, _quality_gate_status
from src.milestone_2.quality_gate.quality_gate import REQUIRED_COLUMNS as QUALITY_GATE_REQUIRED_COLUMNS


def _er_row(
    source_row_id: int,
    month: str,
    *,
    title: str,
    raw_description: str,
    normalized_description: str,
) -> dict:
    row = {name: None for name in RAW_COLUMNS}
    row.update(
        {
            "source_row_id": source_row_id,
            "analysis_month": month,
            "title": title,
            "description": raw_description,
            "title_normalized": title,
            "description_normalized": normalized_description,
            "cat2_slug": "residential-sell",
            "cat3_slug": "apartment-sell",
            "city_slug": "tehran",
            "neighborhood_slug": "test-neighborhood",
            "user_type": "personal",
            "price_value": "5000000000",
            "price_value_toman": 5_000_000_000.0,
            "building_size": "100",
            "building_size_sqm": 100.0,
            "rooms_count": "2",
            "rooms_count_num": 2.0,
            "floor": "2",
            "construction_year": "1400",
            "has_parking": "true",
            "has_elevator": "true",
            "has_warehouse": "true",
            "location_latitude": "35.7",
            "location_longitude": "51.4",
        }
    )
    return row


def test_advanced_entity_resolution_links_candidates_but_only_excludes_safe_same_month_excess(
    tmp_path: Path,
) -> None:
    rows = [
        _er_row(1, "2024-05-01", title="100 sqm apartment", raw_description="A clean unit", normalized_description="clean unit with parking and elevator"),
        _er_row(2, "2024-05-01", title="100 sqm apartment", raw_description="A clean unit!", normalized_description="clean unit with parking and elevator"),
        _er_row(3, "2024-06-01", title="family apartment", raw_description="description alpha", normalized_description="description alpha with distinct words"),
        _er_row(4, "2024-06-01", title="family apartment", raw_description="description beta", normalized_description="description beta with other words"),
        _er_row(5, "2024-07-01", title="100 sqm apartment", raw_description="A clean unit later", normalized_description="clean unit with parking and elevator"),
    ]
    resolved = add_duplicate_features(pl.DataFrame(rows).lazy()).collect()

    high = resolved.filter(pl.col("source_row_id").is_in([1, 2]))
    assert set(high.get_column("deduplication_confidence").to_list()) == {"high"}
    assert high.get_column("probable_duplicate_cluster_id").null_count() == 0
    assert int(high.get_column("same_month_duplicate_excess_flag").sum()) == 1

    medium = resolved.filter(pl.col("source_row_id").is_in([3, 4]))
    assert set(medium.get_column("deduplication_confidence").to_list()) == {"medium"}
    assert medium.get_column("probable_duplicate_cluster_id").null_count() == 0
    assert int(medium.get_column("supply_keep_flag").sum()) == 2

    cross = resolved.filter(pl.col("source_row_id") == 5).row(0, named=True)
    assert cross["cross_month_repeat_flag"] is True
    assert cross["probable_duplicate_cluster_id"] is not None
    assert cross["supply_keep_flag"] is True

    parquet = tmp_path / "silver.parquet"
    resolved.write_parquet(parquet)
    summary_path, _, review_path = write_duplicate_reports(
        parquet, tmp_path / "duplicates", tmp_path / "qa"
    )
    summary = pl.read_csv(summary_path).with_columns(pl.col("value").cast(pl.String))
    metrics = dict(summary.select(["metric", "value"]).iter_rows())
    assert metrics["entity_resolution_method"] == "deterministic_multi_pass_record_linkage"
    assert metrics["entity_resolution_cluster_coverage_rate"] == "1.000000"
    assert metrics["advanced_entity_resolution_bonus_ready"] == "true"
    review = pl.read_csv(review_path)
    assert {"entity_resolution_candidate_flag", "entity_resolution_role"}.issubset(review.columns)


def test_final_sale_psm_and_rent_equivalent_formulas() -> None:
    frame = pl.DataFrame(
        {
            "is_core_analysis_period": [True, True],
            "price_regime": ["sale", "rent_plus_deposit"],
            "sale_price_clean_toman": [1_000_000_000, None],
            "primary_area_sqm": [100.0, 80.0],
            "price_per_sqm_raw_toman": [10_000_000.0, None],
            "outlier_area_flag": [False, False],
            "sale_price_per_sqm_outlier_flag": [False, False],
            "price_regime_conflict_flag": [False, False],
            "type_parse_error_count": [0, 0],
            "monthly_rent_outlier_flag": [False, False],
            "deposit_outlier_flag": [False, False],
            "supply_keep_flag": [True, True],
            "coordinate_pair_present": [True, True],
            "geo_country_valid": [True, True],
            "coordinate_partial_flag": [False, False],
            "monthly_rent_clean_toman": [None, 5_000_000],
            "deposit_clean_toman": [None, 500_000_000],
            "price_regime_review_flag": [False, False],
            "primary_area_missing_flag": [False, False],
            "outlier_price_flag": [False, False],
            "outlier_year_flag": [False, False],
            "exact_duplicate_flag": [False, False],
            "same_month_duplicate_flag": [False, False],
            "area_manual_review_flag": [False, False],
        }
    ).lazy()
    result = apply_final_metrics(frame).collect()
    assert result["sale_price_per_sqm_final_toman"][0] == 10_000_000
    assert result["rent_equivalent_monthly_base_toman"][1] == 20_000_000


def test_outlier_policy_is_latest_flag_only_contract_configuration() -> None:
    assert setting("milestone_2", "versions", "outlier") == "outlier-policy-m2-v2"
    assert float(setting("milestone_2", "outliers", "canonical_iqr_multiplier")) == 3.0
    assert float(setting("milestone_2", "outliers", "sensitivity_iqr_multiplier")) == 2.0
    assert int(setting("milestone_2", "outliers", "min_local_group_rows")) == 50
    assert int(setting("milestone_2", "outliers", "min_fallback_group_rows")) == 200


def test_quality_gate_and_closeout_cover_the_required_m2_contract(tmp_path: Path) -> None:
    for column in {
        "source_row_id",
        "analysis_month",
        "price_regime",
        "probable_duplicate_cluster_id",
        "supply_keep_flag",
        "outlier_price_flag",
        "sale_price_per_sqm_final_toman",
        "rent_equivalent_monthly_base_toman",
        "price_unit",
        "price_observation_type",
    }:
        assert column in QUALITY_GATE_REQUIRED_COLUMNS

    assert {
        "silver_master",
        "standardization_summary",
        "missingness_action_table",
        "duplicate_summary",
        "currency_validation_summary",
        "price_regime_review_summary",
        "outlier_summary",
        "outlier_sensitivity",
        "final_metric_summary",
        "quality_gate_summary",
    }.issubset(REQUIRED_OUTPUTS)
    assert not any("hash" in name.lower() for name in REQUIRED_OUTPUTS)

    gate = tmp_path / "quality_gate_summary.csv"
    pl.DataFrame(
        {
            "check": ["source_row_id_unique", "currency_source_confirmation"],
            "status": ["PASS", "REVIEW"],
            "critical": [True, False],
        }
    ).write_csv(gate)
    critical_failures, review_count = _quality_gate_status(gate)
    assert critical_failures == 0
    assert review_count == 1
