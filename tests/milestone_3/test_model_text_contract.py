from __future__ import annotations

import re

import numpy as np
import pandas as pd

from src.milestone_3.price_drivers.price_drivers import (
    LOG_TARGET,
    TARGET,
    _metrics,
    _prepare,
    build_pipeline,
    error_analysis,
)
from src.milestone_3.seller_type_comparison.seller_type_comparison import (
    AGENCY,
    PERSONAL,
    stratified_comparison,
)
from src.milestone_3.text_analysis.text_price_signals import _benjamini_hochberg
from src.milestone_3.text_analysis.text_rules import KEYWORD_RULES, MANDATORY_ASSIGNMENT_KEYWORDS


def test_sparse_ridge_pipeline_handles_missing_values() -> None:
    n = 120
    frame = pd.DataFrame(
        {
            "primary_area_sqm": np.linspace(50, 180, n),
            "rooms_count_num": np.tile([1, 2, 3, np.nan], n // 4),
            "building_age_years": np.linspace(0, 30, n),
            "floor_num": np.tile([1, 2, 3, 4], n // 4),
            "has_elevator_bool": np.tile([True, False, None, True], n // 4),
            "city_slug": np.tile(["a", "b"], n // 2),
            "neighborhood_slug": np.tile(["n1", "n2", "n3"], n // 3),
            "cat3_slug": "apartment-sell",
            "property_family": "apartment",
            "analysis_month": np.tile(["2024-05", "2024-06", "2024-07"], n // 3),
            "building_direction": np.tile(["north", "south"], n // 2),
        }
    )
    y = np.log(10_000_000 + frame["primary_area_sqm"].to_numpy() * 100_000)
    pipe, features = build_pipeline(frame.columns, 1.0, 2)
    pipe.fit(_prepare(frame, features), y)
    pred = pipe.predict(_prepare(frame.iloc[:10], features))
    assert pred.shape == (10,)
    assert np.isfinite(pred).all()


def test_avm_metrics_and_error_analysis_surface_robust_heldout_error_distribution() -> None:
    actual = np.array([10.0, 20.0, 40.0, 80.0])
    frame = pd.DataFrame({TARGET: actual, LOG_TARGET: np.log(actual)})
    metrics = _metrics(frame, np.log(actual), "test")
    assert np.isclose(metrics["median_absolute_percentage_error_pct"], 0.0)
    assert metrics["within_20pct_share_pct"] == 100.0
    assert metrics["within_30pct_share_pct"] == 100.0
    assert metrics["within_50pct_share_pct"] == 100.0

    error_frame = pd.DataFrame(
        {
            "city_slug": ["tehran"] * 4,
            "property_family": ["apartment"] * 4,
            TARGET: [10.0, 20.0, 40.0, 80.0],
        }
    )
    out = error_analysis(
        error_frame,
        np.log(np.array([10.0, 18.0, 44.0, 72.0])),
        minimum_n=3,
    )
    assert "mean_ape_pct" not in out.columns
    for column in [
        "median_ape_pct",
        "p75_ape_pct",
        "p90_ape_pct",
        "within_20pct_share_pct",
        "within_30pct_share_pct",
        "within_50pct_share_pct",
        "median_abs_error_toman_per_sqm",
        "reliability_status",
    ]:
        assert column in out.columns
    city = out.loc[out["error_scope"] == "city"].iloc[0]
    assert city["reliability_status"] == "RELIABLE"


def test_seller_similar_unit_strata_produces_observational_agency_personal_contrast() -> None:
    rows: list[dict[str, object]] = []
    for seller, log_price in [(AGENCY, np.log(120.0)), (PERSONAL, np.log(100.0))]:
        for i in range(8):
            rows.append(
                {
                    "user_type": seller,
                    LOG_TARGET: log_price,
                    "city_slug": "tehran",
                    "cat3_slug": "apartment-sell",
                    "primary_area_sqm": 100.0,
                    "building_age_years": 10.0,
                    "rooms_count_num": 2.0,
                }
            )
    summary, detail = stratified_comparison(pd.DataFrame(rows), minimum_seller_n=5)
    assert not summary.empty
    assert not detail.empty
    assert summary.loc[0, "method"] == "coarsened_similar_unit_strata"
    assert summary.loc[0, "price_difference_pct_agency_minus_personal"] > 0


def test_text_keyword_contract_contains_mandatory_and_migration_families() -> None:
    assert {"urgent", "below_market"}.issubset(KEYWORD_RULES)
    assert MANDATORY_ASSIGNMENT_KEYWORDS == {"urgent", "below_market"}
    pattern = re.compile(KEYWORD_RULES["migration_sale"]["pattern"])
    examples = [
        "فروش به دلیل مهاجرت",
        "به علت مهاجرت فروش فوری",
        "به خاطر مهاجرت",
        "فروش فوری به دلیل مهاجرت",
        "مهاجرت داریم و فروش ملک",
    ]
    assert all(pattern.search(text) for text in examples)


def test_text_multiple_testing_uses_benjamini_hochberg_fdr() -> None:
    p = pd.Series(
        [0.0001, 0.0002, 0.014, 0.038, 0.116, 0.954],
        index=["a", "b", "c", "d", "e", "f"],
    )
    q = _benjamini_hochberg(p)
    assert np.isfinite(q).all()
    assert (q >= p).all()
    assert q["a"] < 0.05 and q["b"] < 0.05 and q["c"] < 0.05
    assert q["d"] > 0.05 and q["e"] > 0.05 and q["f"] > 0.05
