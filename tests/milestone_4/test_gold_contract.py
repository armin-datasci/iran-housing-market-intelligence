from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from src.milestone_4.gold import build_gold
from src.milestone_4.gold.contracts import (
    COMMON_MART_METADATA_COLUMNS,
    DASHBOARD_PAGES,
    EXPECTED_DIMENSIONS,
    EXPECTED_MARTS,
    EXPECTED_TEXT_SIGNALS,
    MART_GRAIN_KEYS,
    RELATIONSHIP_SPECS,
    SEMANTIC_ONLY_DIMENSIONS,
)


def test_canonical_architecture_is_exact_10_by_5() -> None:
    assert len(EXPECTED_MARTS) == 10
    assert len(EXPECTED_DIMENSIONS) == 5
    assert "mart_location_market" in EXPECTED_MARTS
    assert "mart_text_monthly" in EXPECTED_MARTS
    assert "mart_neighborhood_market" not in EXPECTED_MARTS
    assert "mart_market_temperature" not in EXPECTED_MARTS
    assert "dim_segment" not in EXPECTED_DIMENSIONS
    assert SEMANTIC_ONLY_DIMENSIONS == {"dim_user_type"}
    assert len(RELATIONSHIP_SPECS) == 13


def test_section_29_page_registry_is_exact() -> None:
    assert DASHBOARD_PAGES == {
        "P01": "Executive Market Overview",
        "P02": "Data Quality",
        "P03": "Price Map",
        "P04": "Supply and Price Trends",
        "P05": "Amenities and Price Drivers",
        "P06": "Seller Type Comparison",
        "P07": "Text Signals",
        "P08": "Market Segments",
    }


def test_market_temperature_gate_is_all_city(tmp_path: Path, monkeypatch) -> None:
    source_path = tmp_path / "market_temperature_summary.csv"
    pl.DataFrame(
        {
            "entity_level": ["neighborhood"], "city_slug": ["tabriz"], "neighborhood_slug": ["sample"],
            "price_trend_pct_per_month": [1.2], "supply_trend_pct_per_month": [3.4],
            "price_direction_consistency": [0.8], "supply_direction_consistency": [0.7],
            "trend_stability_score": [0.75], "price_month_count": [6], "supply_month_count": [7],
            "usable_month_count": [6], "sample_size": [250], "median_monthly_listing_count": [42.0],
            "evidence_multiplier": [0.95], "price_rank": [0.9], "supply_signal_rank": [0.8],
            "market_temperature_score": [72.0], "market_temperature_percentile": [0.92],
            "market_temperature_label": ["HOT"], "market_temperature_rank": [4.0],
            "proxy_definition": ["Listing-Market Temperature Proxy"], "supply_population": ["apartment_sale"],
            "price_population": ["apartment_sale"], "listing_activity_interpretation": ["platform listing activity"],
            "analysis_version": ["m3-market-temperature-v1.3-frozen-final"],
        }
    ).write_csv(source_path)
    monkeypatch.setitem(build_gold.SOURCES, "market_temperature", source_path)
    manifest = {"definition": {"professor_facing_reliability_gate": {
        "minimum_reliable_price_months": 5, "minimum_listing_activity_months": 5, "minimum_sample_size": 100,
    }}}
    out = build_gold.build_market_temperature(manifest)
    assert out.item(0, "temperature_reliability_eligible_flag") is True
    assert out.item(0, "professor_facing_eligible_flag") is True


def test_location_mart_merges_price_and_temperature(tmp_path: Path, monkeypatch) -> None:
    map_path = tmp_path / "neighborhood_market_summary.csv"
    temp_path = tmp_path / "market_temperature_summary.csv"
    pl.DataFrame({
        "city_slug": ["tehran"], "neighborhood_slug": ["sample"], "listing_n": [100],
        "median_asking_price_per_sqm_toman": [50_000_000.0], "p25_asking_price_per_sqm_toman": [45_000_000.0],
        "p75_asking_price_per_sqm_toman": [55_000_000.0], "iqr_asking_price_per_sqm_toman": [10_000_000.0],
        "reliable_flag": [True], "reliability_status": ["PASS"], "reliability_reason": ["N>=30"],
        "metric_definition": ["apartment_sale_asking_price_per_sqm"], "analysis_version": ["map-v1"],
    }).write_csv(map_path)
    pl.DataFrame({
        "entity_level": ["neighborhood", "city"], "city_slug": ["tehran", "tehran"],
        "neighborhood_slug": ["sample", None], "price_trend_pct_per_month": [1.0, 0.5],
        "supply_trend_pct_per_month": [2.0, 1.0], "price_direction_consistency": [0.8, 0.8],
        "supply_direction_consistency": [0.7, 0.7], "trend_stability_score": [0.75, 0.75],
        "price_month_count": [6, 6], "supply_month_count": [6, 6], "usable_month_count": [6, 6],
        "sample_size": [200, 1000], "median_monthly_listing_count": [40.0, 300.0], "evidence_multiplier": [0.9, 0.9],
        "price_rank": [0.8, 0.6], "supply_signal_rank": [0.7, 0.5], "market_temperature_score": [70.0, 55.0],
        "market_temperature_percentile": [0.9, 0.6], "market_temperature_label": ["HOT", "NEUTRAL"],
        "market_temperature_rank": [1.0, 1.0], "proxy_definition": ["proxy", "proxy"],
        "supply_population": ["apartment_sale", "apartment_sale"], "price_population": ["apartment_sale", "apartment_sale"],
        "listing_activity_interpretation": ["platform", "platform"], "analysis_version": ["temp-v1", "temp-v1"],
    }).write_csv(temp_path)
    monkeypatch.setitem(build_gold.SOURCES, "neighborhood_market", map_path)
    monkeypatch.setitem(build_gold.SOURCES, "market_temperature", temp_path)
    gate = {"definition": {"professor_facing_reliability_gate": {
        "minimum_reliable_price_months": 5, "minimum_listing_activity_months": 5, "minimum_sample_size": 100,
    }}}
    out = build_gold.build_location_market({"minimum_valid_listings": 30}, gate)
    assert out.height == 2
    n = out.filter(pl.col("entity_level") == "neighborhood")
    assert n.item(0, "price_map_available_flag") is True
    assert n.item(0, "temperature_available_flag") is True
    assert n.item(0, "median_asking_price_per_sqm_toman") == 50_000_000.0
    assert n.item(0, "market_temperature_score") == 70.0


def test_relationship_contract_only_contains_active_physical_rows() -> None:
    contract = build_gold.build_gold_relationship_contract()
    assert contract.height == 13
    assert contract.get_column("active").all()
    assert set(contract.get_column("cross_filter_direction").to_list()) == {"Single"}
    assert set(contract.get_column("cardinality").to_list()) == {"1:*"}
    assert "dim_user_type" not in set(contract.get_column("from_table").to_list())


def test_text_signal_contract_is_six_families() -> None:
    assert EXPECTED_TEXT_SIGNALS == {"new_build", "unused", "urgent", "exchange", "below_market", "migration_sale"}
    assert MART_GRAIN_KEYS["mart_text_monthly"] == ("month_key", "keyword_family")


def test_text_monthly_adapts_current_m3_schema(tmp_path: Path, monkeypatch) -> None:
    source_path = tmp_path / "text_keyword_monthly_frequency.csv"
    pl.DataFrame(
        {
            "analysis_month": ["2024-05", "2024-06"],
            "keyword": ["new_build", "new_build"],
            "keyword_fa": ["نوساز", "نوساز"],
            "positive_n": [3562, 3339],
            "population_n": [48839, 48801],
            "positive_rate": [0.072934, 0.068421],
        }
    ).write_csv(source_path)
    monkeypatch.setitem(build_gold.SOURCES, "text_monthly", source_path)

    out = build_gold.build_text_monthly("text-v-test")

    assert out.get_column("sale_valid_listing_count").to_list() == [48839, 48801]
    assert out.get_column("matched_listing_count").to_list() == [3562, 3339]
    assert out.get_column("matched_rate").to_list() == [0.072934, 0.068421]
    assert out.get_column("matched_rate_pct").to_list() == pytest.approx([7.2934, 6.8421], abs=1e-10)
    assert out.get_column("keyword_family").to_list() == ["new_build", "new_build"]


def test_segment_positioning_reconstructs_city_cat3_reference_from_current_handoff(tmp_path: Path, monkeypatch) -> None:
    assignments_path = tmp_path / "listing_segments.parquet"
    features_path = tmp_path / "analysis_ready_features.parquet"
    pl.DataFrame(
        {
            "source_row_id": [1, 2, 3, 4, 5, 6],
            "segment_id": ["SEG01", "SEG01", "SEG02", "SEG02", "SEG02", "SEG04"],
            "city_slug": ["tehran", "tehran", "karaj", "tehran", "karaj", "tehran"],
            "cat3_slug": ["apartment-sell", "apartment-sell", "apartment-sell", "villa-sell", "villa-sell", "plot-old"],
        }
    ).write_parquet(assignments_path)
    pl.DataFrame(
        {
            "source_row_id": [1, 2, 3, 4, 5, 6],
            "sale_price_per_sqm_final_toman": [100.0, 200.0, 300.0, 400.0, 500.0, 600.0],
        }
    ).write_parquet(features_path)
    monkeypatch.setitem(build_gold.SOURCES, "segment_assignments", assignments_path)
    monkeypatch.setitem(build_gold.SOURCES, "analysis_ready_features", features_path)
    monkeypatch.setattr(build_gold, "setting", lambda *args, **kwargs: 2)

    out = build_gold._segment_positioning().sort("segment_id")

    assert out.get_column("segment_id").to_list() == ["SEG01", "SEG02", "SEG04"]
    assert out.get_column("positioning_listing_n").to_list() == [2, 3, 1]
    assert out.get_column("median_normalized_price_index").to_list() == pytest.approx(
        [100.0, 111.11111111111111, 171.42857142857142], abs=1e-10
    )


def test_standardize_mart_has_common_metadata() -> None:
    frame = pl.DataFrame({
        "feature_id": ["x"], "feature_display_name": ["X"], "adjusted_effect_pct": [1.0],
        "ci_lower_pct": [0.0], "ci_upper_pct": [2.0], "sample_n": [100], "method_version": ["v1"],
    })
    out = build_gold.standardize_mart("mart_price_driver_effects", frame)
    assert set(COMMON_MART_METADATA_COLUMNS).issubset(out.columns)
    assert out.item(0, "sample_n") == 100



def test_location_dimension_uses_explicit_schema_after_many_null_neighborhoods() -> None:
    city_rows = 101
    frame = pl.DataFrame(
        {
            "location_key": [f"CITY:city-{i:03d}" for i in range(city_rows)]
            + ["NEIGHBORHOOD:ahvaz|amaniyeh-ahvaz"],
            "city_slug": [f"city-{i:03d}" for i in range(city_rows)] + ["ahvaz"],
            "neighborhood_slug": [None] * city_rows + ["amaniyeh-ahvaz"],
        }
    )

    out = build_gold.build_location_dimension(frame)

    assert out.height == city_rows + 1
    assert out.schema["neighborhood_slug"] == pl.String
    neighborhood = out.filter(pl.col("neighborhood_slug") == "amaniyeh-ahvaz")
    assert neighborhood.height == 1
    assert neighborhood.item(0, "location_level") == "neighborhood"
    assert neighborhood.item(0, "city_slug") == "ahvaz"
    assert neighborhood.item(0, "city_key") == "CITY:ahvaz"



def test_month_dimension_unions_all_month_sensitive_marts() -> None:
    a = pl.DataFrame({"analysis_month": ["2024-05"], "month_key": [202405]})
    b = pl.DataFrame({"analysis_month": ["2024-06"], "month_key": [202406]})
    c = pl.DataFrame({"analysis_month": ["2024-07"], "month_key": [202407]})
    out = build_gold.build_month_dimension(a, b, c)
    assert out.get_column("month_key").to_list() == [202405, 202406, 202407]


def test_sources_exclude_legacy_downstream_artifacts() -> None:
    text = "\n".join(str(path).lower() for path in build_gold.SOURCES.values())
    for forbidden in ["top_5_hot", "top_5_cold", "neighborhood_extremes", "statistical_tests", "seller_type_segment_analysis"]:
        assert forbidden not in text
    assert build_gold.SOURCES["text_monthly"].as_posix().endswith("text_analysis/text_keyword_monthly_frequency.csv")
    assert "segment_monthly" not in build_gold.SOURCES

def test_market_monthly_excludes_unmapped_city_bucket_without_relabeling(
    tmp_path: Path, monkeypatch
) -> None:
    market_path = tmp_path / "monthly_market_summary.csv"
    supply_path = tmp_path / "monthly_supply_summary.csv"

    pl.DataFrame(
        {
            "analysis_month": ["2024-06-01"],
            "entity_level": ["city"],
            "city_slug": ["tehran"],
            "neighborhood_slug": [None],
            "raw_listing_count": [100],
            "deduplicated_listing_count": [99],
            "deduplicated_supply_mom_pct": [1.0],
            "price_listing_n": [80],
            "median_asking_price_per_sqm_toman": [50_000_000.0],
            "median_price_mom_pct": [2.0],
            "price_reliable_flag": [True],
            "market_scope": ["apartment_sale_proxy"],
            "analysis_version": ["monthly-v-test"],
        }
    ).write_csv(market_path)

    # The null-city row is an upstream aggregate bucket for listings whose city is
    # missing. The national row already preserves those listings, so Gold must not
    # invent a city or relabel the bucket as NATIONAL.
    pl.DataFrame(
        {
            "analysis_month": ["2024-06-01", "2024-06-01", "2024-06-01"],
            "entity_level": ["city", "city", "national"],
            "city_slug": [None, "tehran", None],
            "neighborhood_slug": [None, None, None],
            "market_scope": ["all_market", "all_market", "all_market"],
            "raw_listing_count": [2, 10, 12],
            "deduplicated_listing_count": [2, 10, 12],
            "deduplicated_supply_mom_pct": [None, None, None],
            "duplicate_supply_reduction_rate": [0.0, 0.0, 0.0],
        }
    ).write_csv(supply_path)

    monkeypatch.setitem(build_gold.SOURCES, "monthly_market", market_path)
    monkeypatch.setitem(build_gold.SOURCES, "monthly_supply", supply_path)

    out = build_gold.build_market_monthly("monthly-v-test")

    assert out.get_column("location_key").null_count() == 0
    assert out.filter(
        (pl.col("entity_level") == "city") & pl.col("city_slug").is_null()
    ).height == 0
    assert out.filter(
        (pl.col("series_kind") == "supply_only")
        & (pl.col("market_scope") == "all_market")
    ).select("raw_listing_count").get_column("raw_listing_count").sort().to_list() == [10, 12]
    national = out.filter(
        (pl.col("series_kind") == "supply_only")
        & (pl.col("entity_level") == "national")
        & (pl.col("market_scope") == "all_market")
    )
    assert national.height == 1
    assert national.item(0, "location_key") == "NATIONAL"
    assert national.item(0, "raw_listing_count") == 12


def test_m2_quality_summary_adapter_separates_review_from_readiness(
    tmp_path: Path
) -> None:
    path = tmp_path / "quality_gate_summary.csv"
    pl.DataFrame(
        {
            "check": [
                "source_row_id_unique",
                "sale_psm_eligible_rows",
                "currency_source_confirmation",
            ],
            "actual": ["1000000", "405809", "toman_assumed_unconfirmed"],
            "expected": ["1000000", ">0", "toman_assumed_unconfirmed"],
            "status": ["PASS", "PASS", "REVIEW"],
            "critical": [True, True, False],
            "notes": ["", "", "documented limitation"],
        }
    ).write_csv(path)

    row = build_gold._tabular_quality_status_row(
        "M2", "Price Quality Gate", path
    )

    assert row["overall_status"] == "REVIEW"
    assert row["ready"] is True
    assert row["critical_failures"] == 0
    assert row["review_count"] == 1
    assert row["reliability_status"] == "review"
    assert row["sample_n"] == 1_000_000



def test_power_bi_relationship_rows_are_all_active_single_direction_one_to_many() -> None:
    from src.milestone_4.gold.contracts import relationship_contract_rows

    rows = relationship_contract_rows()
    assert len(rows) == 13
    assert all(row["relationship_type"] == "physical_one_to_many" for row in rows)
    assert all(row["cardinality"] == "1:*" for row in rows)
    assert all(row["cross_filter_direction"] == "Single" for row in rows)
    assert all(row["active"] is True for row in rows)
    assert all(row["expected_unique_one_side"] is True for row in rows)
