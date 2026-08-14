from __future__ import annotations

import csv
from pathlib import Path

import yaml

from src.milestone_1.data_loading.data_loading import inspect_raw_data
from src.milestone_1.documentation.data_documentation import (
    REQUIRED_FIELDS,
    build_contract_from_dictionary,
)
from src.milestone_1.milestone1_closeout import AGREED_M1_OUTPUTS


def test_m1_closeout_targets_only_canonical_deliverables() -> None:
    assert set(AGREED_M1_OUTPUTS) == {
        "data_contract",
        "data_dictionary",
        "data_audit_report",
    }


def test_raw_inspection_is_full_schema_read_without_persisted_dev_sample(tmp_path: Path) -> None:
    raw = tmp_path / "raw.csv"
    raw.write_text("city_slug,price_value\ntehran,100\nkaraj,200\n", encoding="utf-8")

    result = inspect_raw_data(raw, expected_columns=2, expected_rows=2)

    assert result["row_count"] == 2
    assert result["column_count"] == 2
    assert result["columns"] == ["city_slug", "price_value"]
    assert result["status"]["ready"] is True
    assert list(tmp_path.glob("*.parquet")) == []


def test_data_contract_builder_preserves_required_documentation_fields(tmp_path: Path) -> None:
    dictionary = tmp_path / "data_dictionary.csv"
    contract = tmp_path / "data_contract.yaml"
    with dictionary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REQUIRED_FIELDS)
        writer.writeheader()
        writer.writerow(
            {
                "column_name": "city_slug",
                "business_meaning": "Canonical city identifier",
                "raw_dtype": "String",
                "expected_dtype": "String",
                "unit": "",
                "allowed_values": "documented source domain",
                "missing_meaning": "unknown",
                "cleaning_rule": "preserve/standardize",
                "quality_risk": "missing geography",
                "owner": "M1/M2",
            }
        )

    build_contract_from_dictionary(dictionary_path=dictionary, contract_path=contract)
    payload = yaml.safe_load(contract.read_text(encoding="utf-8"))

    assert payload["stage"] == "milestone_1_source_contract"
    assert payload["global_assumptions"]["raw_is_read_only"] is True
    assert payload["global_assumptions"]["no_blanket_zero_imputation"] is True
    assert payload["global_assumptions"]["price_observation_type"] == "asking_price"
    assert payload["required_column_metadata"] == REQUIRED_FIELDS
