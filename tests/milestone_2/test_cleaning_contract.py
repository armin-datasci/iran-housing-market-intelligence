from __future__ import annotations

from pathlib import Path

import pandas as pd
import polars as pl
import pytest

from src.common.config import price_observation_type, price_unit, setting
from src.final_reporting.build_technical_report import _standardization_section
from src.milestone_2.cleaning_management.missingness.missingness_audit import _action_for
from src.milestone_2.cleaning_management.standardization.structured_standardization import (
    normalized_number_text,
)
from src.milestone_2.cleaning_management.standardization.text_standardization import (
    add_normalized_text_columns,
)
from src.milestone_2.price_regimes.price_regimes import PRICE_REGIME_DOMAIN, apply_price_regimes


EXPECTED_PRICE_REGIMES = [
    "sale",
    "rent_plus_deposit",
    "full_deposit",
    "rent_only",
    "rent_negotiable",
    "rent_unknown_or_incomplete",
    "temporary_rent",
    "service",
    "unknown",
]


def test_persian_text_normalization_preserves_raw_and_normalizes_characters() -> None:
    frame = pl.DataFrame(
        {"title": ["  خانه يک ۱۲۳  "], "description": ["متن\u00a0آگهی"]}
    ).lazy()
    result = add_normalized_text_columns(frame).collect()
    assert result["title"][0] == "  خانه يک ۱۲۳  "
    assert result["title_normalized"][0] == "خانه یک 123"
    assert result["description_normalized"][0] == "متن آگهی"


def test_structured_numeric_normalization_handles_persian_arabic_digits_and_separators() -> None:
    frame = pl.DataFrame({"value": [" ۱۲۳٬۴۵۶ ", "١٢٣,٤٥٦", None]})
    out = frame.select(normalized_number_text("value").alias("normalized"))
    assert out.get_column("normalized").to_list() == ["123456", "123456", None]


def test_missingness_policy_never_blanket_fills_unknown_with_zero_or_false() -> None:
    bool_action, _, bool_effect = _action_for("has_parking_bool", 0.25)
    numeric_action, numeric_reason, _ = _action_for("primary_area_sqm", 0.25)
    assert "do not convert unknown amenities to False" in bool_action
    assert "bias" in bool_effect.lower()
    assert "Preserve null" in numeric_action
    assert "not equivalent to zero" in numeric_reason


def test_canonical_currency_and_observation_contract_is_not_silently_converted() -> None:
    assert price_unit() == "toman_assumed_unconfirmed"
    assert price_observation_type() == "asking_price"
    assert setting("contracts", "factor_of_ten_conversion_applied") is False


def test_price_regime_domain_is_exact_nine_states_and_semantic_zero_is_retained() -> None:
    assert PRICE_REGIME_DOMAIN == EXPECTED_PRICE_REGIMES
    frame = pl.DataFrame(
        {
            "market_regime": ["long_term_rent", "long_term_rent", "long_term_rent"],
            "long_term_rent_regime": ["rent_plus_deposit", "full_deposit", "rent_only"],
            "sale_price_status": ["not_applicable"] * 3,
            "rent_component_status": ["fixed_positive", "semantic_zero", "fixed_positive"],
            "credit_component_status": ["fixed_positive", "fixed_positive", "semantic_zero"],
            "price_mode": [None] * 3,
            "price_value_toman": [None] * 3,
            "rent_mode": ["مقطوع", "مجانی", "مقطوع"],
            "rent_value_toman": [10, 0, 10],
            "credit_mode": ["مقطوع", "مقطوع", "مجانی"],
            "credit_value_toman": [100, 100, 0],
            "is_sale": [False] * 3,
            "is_long_term_rent": [True] * 3,
        }
    ).lazy()
    result = apply_price_regimes(frame).collect()
    assert result["price_regime"].to_list() == [
        "rent_plus_deposit",
        "full_deposit",
        "rent_only",
    ]
    assert result["monthly_rent_clean_toman"].to_list() == [10, 0, 10]
    assert result["deposit_clean_toman"].to_list() == [100, 100, 0]


def _write_standardization_summary(path: Path, *, include_all: bool = True) -> None:
    rows = [
        {"metric": "row_count", "value": "1000000", "status": "PASS"},
        {"metric": "rows_with_parse_error", "value": "724", "status": "REVIEW"},
        {"metric": "analysis_month_parse_failure_rows", "value": "0", "status": "PASS"},
        {"metric": "title_normalized_changed_rows", "value": "516041", "status": "PASS"},
        {"metric": "description_normalized_changed_rows", "value": "887874", "status": "PASS"},
    ]
    if not include_all:
        rows = [row for row in rows if row["metric"] != "title_normalized_changed_rows"]
    pd.DataFrame(rows).to_csv(path, index=False)


def test_final_technical_report_requires_complete_standardization_evidence(tmp_path: Path) -> None:
    summary = tmp_path / "standardization_summary.csv"
    _write_standardization_summary(summary)
    text, sources, row_count = _standardization_section(tmp_path, summary)
    assert row_count == 1_000_000
    assert "Arabic Yeh/Kaf" in text
    assert "ASCII/Western digits 0-9" in text
    assert "do not overwrite the raw text" in text
    assert "type_parse_error_count" in text
    assert sources == ["standardization_summary.csv"]

    _write_standardization_summary(summary, include_all=False)
    with pytest.raises(RuntimeError, match="missing required metrics"):
        _standardization_section(tmp_path, summary)
