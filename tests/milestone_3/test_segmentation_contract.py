from __future__ import annotations

import numpy as np
import pandas as pd

from src.milestone_3.market_segmentation.market_segmentation import (
    _apartment_tier_name_map,
    _assign_compatible_domain_segments,
    fallback_typology,
    fit_reference,
)


def test_local_reference_and_fallback_typology_are_stable_and_method_honest() -> None:
    n = 120
    frame = pd.DataFrame(
        {
            "source_row_id": range(n),
            "city_slug": np.where(np.arange(n) < 60, "a", "b"),
            "neighborhood_slug": np.where(np.arange(n) % 2 == 0, "n1", "n2"),
            "sale_price_per_sqm_final_toman": np.linspace(10_000_000, 30_000_000, n),
            "primary_area_sqm": np.linspace(50, 180, n),
        }
    )
    neighborhood, city, global_ref = fit_reference(frame, 10)
    assert global_ref > 0
    assert not neighborhood.empty and not city.empty
    segment_id, segment_name, diagnostics = fallback_typology(frame, 10)
    assert segment_id.notna().all()
    assert segment_name.nunique() >= 2
    assert set(segment_id.astype(str).unique()) == {"SEG01", "SEG02", "SEG03"}
    assert diagnostics.iloc[0]["specification"] == "rule_based_fallback"


def test_compatible_property_family_segment_ids_are_stable() -> None:
    frame = pd.DataFrame(
        {
            "property_family": ["land", "commercial", "house"],
            "segment_id": [pd.NA, pd.NA, pd.NA],
            "segment_name": [pd.NA, pd.NA, pd.NA],
            "segment_method": [pd.NA, pd.NA, pd.NA],
        }
    )
    out = _assign_compatible_domain_segments(frame)
    assert out.loc[0, "segment_id"] == "SEG04"
    assert out.loc[1, "segment_id"] == "SEG05"
    assert str(out.loc[2, "segment_id"]).startswith("DOMAIN_")


def test_apartment_semantic_labels_keep_relative_luxury_interpretation() -> None:
    frame = pd.DataFrame(
        {
            "primary_area_sqm": np.r_[np.repeat(65.0, 30), np.repeat(90.0, 30), np.repeat(135.0, 30)],
            "building_age_years": np.r_[np.repeat(15.0, 30), np.repeat(8.0, 30), np.repeat(2.0, 30)],
            "rooms_count_num": np.r_[np.repeat(1.0, 30), np.repeat(2.0, 30), np.repeat(3.0, 30)],
            "relative_price_log": np.r_[np.repeat(-0.25, 30), np.repeat(0.0, 30), np.repeat(0.35, 30)],
            "has_elevator_bool": np.r_[np.repeat(0.2, 30), np.repeat(0.7, 30), np.repeat(1.0, 30)],
            "has_parking_bool": np.r_[np.repeat(0.4, 30), np.repeat(0.8, 30), np.repeat(1.0, 30)],
            "has_warehouse_bool": np.r_[np.repeat(0.5, 30), np.repeat(0.8, 30), np.repeat(0.95, 30)],
            "has_balcony_bool": np.r_[np.repeat(0.6, 30), np.repeat(0.9, 30), np.repeat(1.0, 30)],
        }
    )
    tier = pd.Series(np.repeat(["economic", "mid_market", "luxury"], 30), dtype="string")
    names = _apartment_tier_name_map(frame, tier)
    assert names["economic"].startswith("اقتصادی نسبی")
    assert names["mid_market"].startswith("میان‌رده")
    assert names["luxury"] == "لوکس نسبی | خانوادگی نوساز/کم‌سن امکانات‌دار"
