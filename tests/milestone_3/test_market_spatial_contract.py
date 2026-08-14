from __future__ import annotations

from pathlib import Path

import pandas as pd
import polars as pl
import pytest

from src.common.config import configured_path
from src.milestone_3.analysis_populations.analysis_populations import REQUIRED
from src.milestone_3.market_map.market_map import _write_interactive_city_map
from src.milestone_3.market_temperature.market_temperature import (
    _professor_facing_neighborhoods,
    _score,
    build_features,
)
from src.milestone_3.milestone3_closeout import EXPECTED_OUTPUTS
from src.milestone_3.spatial_quality.spatial_quality import (
    _advanced_spatial_bonus_ready,
    _reverse_core_validation_pass,
    _summarize_reverse_detail,
)


def test_m2_to_m3_silver_schema_contract_if_available() -> None:
    silver: Path = configured_path("silver_master")
    if not silver.exists():
        pytest.skip("Canonical Silver Master is not available in this test environment.")
    columns = set(pl.scan_parquet(silver).collect_schema().names())
    missing = sorted(REQUIRED - columns)
    assert not missing, f"Silver Master is missing M3-required columns: {missing}"


def test_m3_closeout_contains_only_canonical_analytical_outputs() -> None:
    names = set(EXPECTED_OUTPUTS)
    assert {
        "analysis_population_summary",
        "spatial_quality_summary",
        "city_market_summary",
        "neighborhood_market_summary",
        "monthly_market_summary",
        "market_temperature_summary",
        "price_driver_summary",
        "seller_type_comparison_summary",
        "text_signal_summary",
        "segment_profile",
        "listing_segments",
    }.issubset(names)
    assert all("manifest" not in name for name in names)


def test_interactive_city_map_uses_only_rounded_aggregate_coordinates(tmp_path: Path) -> None:
    output = tmp_path / "four_city_market_map.html"
    rows = [
        {
            "city_slug": "tehran",
            "city_latitude": 35.712345,
            "city_longitude": 51.401234,
            "median_asking_price_per_sqm_toman": 81_000_000,
            "listing_n": 12_000,
            "reliable_neighborhood_n": 145,
        },
        {
            "city_slug": "mashhad",
            "city_latitude": 36.287654,
            "city_longitude": 59.611234,
            "median_asking_price_per_sqm_toman": 32_100_000,
            "listing_n": 6_000,
            "reliable_neighborhood_n": 62,
        },
    ]
    assert _write_interactive_city_map(rows, output, min_n=30)
    html = output.read_text(encoding="utf-8")
    assert "Tehran" in html and "Mashhad" in html
    assert "Offline self-contained artifact" in html
    assert "leaflet.js" not in html
    assert "tile.openstreetmap.org" not in html
    assert "unpkg.com" not in html
    assert "35.712345" not in html and "51.401234" not in html
    assert "35.71" in html and "51.4" in html
    assert "source_row_id" not in html


def _monthly_rows(city: str, prices: list[int], listings: list[int]) -> list[dict[str, object]]:
    months = ["2024-05", "2024-06", "2024-07", "2024-08"]
    return [
        {
            "analysis_month": month,
            "entity_level": "city",
            "city_slug": city,
            "neighborhood_slug": None,
            "deduplicated_listing_count": supply,
            "price_listing_n": max(30, int(supply)),
            "median_asking_price_per_sqm_toman": price,
            "price_reliable_flag": True,
        }
        for month, price, supply in zip(months, prices, listings)
    ]


def test_market_temperature_direction_activity_and_evidence_contract() -> None:
    rows = []
    rows += _monthly_rows("rising_both", [100, 110, 120, 130], [100, 110, 120, 130])
    rows += _monthly_rows("falling_both", [130, 120, 110, 100], [130, 120, 110, 100])
    features = build_features(pd.DataFrame(rows), 3)
    scored = _score(
        features,
        0.60,
        0.40,
        0.50,
        0.50,
        "base",
        supply_signal="activity",
        gate_mode="price_only",
    )
    assert scored.loc[scored.city_slug == "rising_both"].iloc[0]["temperature_label_base"] == "HOT"
    assert scored.loc[scored.city_slug == "falling_both"].iloc[0]["temperature_label_base"] == "COLD"

    stable = build_features(
        pd.DataFrame(
            _monthly_rows("stable", [100, 110, 120, 130], [100, 110, 120, 130])
            + _monthly_rows("less_stable", [100, 120, 110, 130], [100, 120, 110, 130])
        ),
        3,
    ).set_index("city_slug")
    assert 0.0 < stable.loc["stable", "evidence_multiplier"] <= 1.0
    assert stable.loc["stable", "evidence_multiplier"] >= stable.loc["less_stable", "evidence_multiplier"]


def test_professor_facing_temperature_gate_requires_history_n_and_four_city_scope() -> None:
    frame = pd.DataFrame(
        [
            {"entity_level": "neighborhood", "city_slug": "tehran", "neighborhood_slug": "reliable", "price_month_count": 8, "supply_month_count": 8, "sample_size": 250},
            {"entity_level": "neighborhood", "city_slug": "tehran", "neighborhood_slug": "short_history", "price_month_count": 4, "supply_month_count": 8, "sample_size": 250},
            {"entity_level": "neighborhood", "city_slug": "karaj", "neighborhood_slug": "small_n", "price_month_count": 8, "supply_month_count": 8, "sample_size": 99},
            {"entity_level": "neighborhood", "city_slug": "shiraz", "neighborhood_slug": "outside_four_city_view", "price_month_count": 8, "supply_month_count": 8, "sample_size": 500},
        ]
    )
    surfaced = _professor_facing_neighborhoods(frame, minimum_months=5, minimum_sample_size=100)
    assert surfaced["neighborhood_slug"].tolist() == ["reliable"]


def _reverse_rows(locality_status: str = "REVIEW") -> list[dict[str, object]]:
    return [
        {"metric": "reverse_geocode_sample_n", "status": "PASS"},
        {"metric": "reverse_geocode_request_success_rate", "status": "PASS"},
        {"metric": "reverse_geocode_iran_country_match_rate", "status": "PASS"},
        {"metric": "reverse_geocode_city_or_county_match_rate", "status": locality_status},
    ]


def test_advanced_spatial_bonus_keeps_locality_taxonomy_review_separate_from_integrity_gate() -> None:
    rows = _reverse_rows("REVIEW")
    assert _reverse_core_validation_pass(rows) is True
    assert _advanced_spatial_bonus_ready(
        boundary_status="PASS", reverse_rows=rows, reverse_requested=True
    ) is True

    bad = _reverse_rows("PASS")
    bad[2]["status"] = "REVIEW"
    assert _reverse_core_validation_pass(bad) is False
    assert _advanced_spatial_bonus_ready(
        boundary_status="PASS", reverse_rows=bad, reverse_requested=True
    ) is False


def test_sanitized_reverse_replay_preserves_review_without_lowering_bonus_threshold() -> None:
    detail = pl.DataFrame(
        {
            "request_status": ["OK", "OK", "OK", "OK"],
            "country_match": [True, True, True, True],
            "city_or_county_match": [True, False, True, False],
        }
    )
    rows, overall = _summarize_reverse_detail(detail)
    assert overall == "REVIEW"
    assert _reverse_core_validation_pass(rows) is True
    assert _advanced_spatial_bonus_ready(
        boundary_status="PASS", reverse_rows=rows, reverse_requested=True
    ) is True
