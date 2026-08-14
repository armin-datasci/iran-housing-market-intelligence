from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import polars as pl
import yaml

from src.common.config import configured_path
from src.common.io_utils import atomic_write_csv, atomic_write_text
from src.common.paths import CONFIG_DIR, DOCS_DIR
from src.common.validation import Check, make_check, summarize_checks

REQUIRED_FIELDS = [
    "column_name", "business_meaning", "raw_dtype", "expected_dtype", "unit",
    "allowed_values", "missing_meaning", "cleaning_rule", "quality_risk", "owner",
]

# M1 validates Raw-schema documentation coverage. In the final repository the shared
# Data Dictionary/Data Contract may already have been promoted to the canonical M2
# Silver release and therefore legitimately contain Raw + derived columns.
ACCEPTED_CONTRACT_STAGES = {
    "milestone_1_source_contract",
    "milestone_2_canonical_release",
}


def source_columns(raw_path: Path) -> list[str]:
    return (
        pl.scan_csv(
            raw_path,
            has_header=True,
            infer_schema=False,
            try_parse_dates=False,
        )
        .collect_schema()
        .names()
    )


def load_dictionary(path: Path = DOCS_DIR / "data_dictionary.csv") -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [{key: str(row.get(key, "")) for key in REQUIRED_FIELDS} for row in rows]


def validate_m1_documentation(
    raw_path: Path | None = None,
    *,
    dictionary_path: Path = DOCS_DIR / "data_dictionary.csv",
    contract_path: Path = CONFIG_DIR / "data_contract.yaml",
) -> tuple[list[Check], dict[str, Any]]:
    """Validate that shared canonical documentation fully covers the M1 Raw schema.

    The final project intentionally has one shared Data Dictionary/Data Contract. Once
    M2 is canonical, those artifacts are supersets of the 60-column Raw schema rather
    than M1-only 60-row files. M1 therefore checks *coverage*, uniqueness, completeness,
    and a supported contract stage; it must not reject legitimate M2-derived columns.
    """
    raw = (raw_path or configured_path("raw_source")).resolve()
    if not raw.exists():
        raise FileNotFoundError(f"Raw source not found: {raw}")
    if not dictionary_path.exists():
        raise FileNotFoundError(f"Data Dictionary not found: {dictionary_path}")
    if not contract_path.exists():
        raise FileNotFoundError(f"Data Contract not found: {contract_path}")

    rows = load_dictionary(dictionary_path)
    dictionary_names = [row["column_name"].strip() for row in rows]
    raw_names = source_columns(raw)

    contract = yaml.safe_load(contract_path.read_text(encoding="utf-8")) or {}
    contract_rows = contract.get("columns", []) or []
    contract_names = [
        str(row.get("column_name", "")).strip()
        for row in contract_rows
        if isinstance(row, dict)
    ]

    raw_set = set(raw_names)
    dictionary_set = set(dictionary_names)
    contract_set = set(contract_names)

    missing_raw_in_dictionary = sorted(raw_set - dictionary_set)
    missing_raw_in_contract = sorted(raw_set - contract_set)
    dictionary_extra_columns = sorted(dictionary_set - raw_set)
    contract_extra_columns = sorted(contract_set - raw_set)
    blank_cells = sum(
        not row[field].strip()
        for row in rows
        for field in REQUIRED_FIELDS
    )
    contract_stage = str(contract.get("stage") or "").strip()

    checks = [
        make_check(
            "dictionary_field_count",
            "documentation",
            len(REQUIRED_FIELDS),
            10,
            len(REQUIRED_FIELDS) == 10,
        ),
        make_check(
            "dictionary_covers_raw_column_count",
            "documentation",
            len(dictionary_names),
            f">={len(raw_names)}",
            len(dictionary_names) >= len(raw_names),
            notes=(
                "The shared canonical dictionary may contain M2-derived fields in addition "
                "to the M1 Raw schema."
            ),
        ),
        make_check(
            "dictionary_covers_raw_schema",
            "documentation",
            missing_raw_in_dictionary,
            [],
            not missing_raw_in_dictionary,
            notes=f"canonical_extra_columns={len(dictionary_extra_columns)}",
        ),
        make_check(
            "dictionary_unique_names",
            "documentation",
            len(dictionary_set),
            len(dictionary_names),
            len(dictionary_set) == len(dictionary_names),
        ),
        make_check(
            "dictionary_no_blank_required_cells",
            "documentation",
            blank_cells,
            0,
            blank_cells == 0,
        ),
        make_check(
            "contract_covers_raw_schema",
            "documentation",
            missing_raw_in_contract,
            [],
            not missing_raw_in_contract,
            notes=f"canonical_extra_columns={len(contract_extra_columns)}",
        ),
        make_check(
            "dictionary_contract_column_alignment",
            "documentation",
            sorted(dictionary_set ^ contract_set),
            [],
            dictionary_set == contract_set,
        ),
        make_check(
            "contract_stage_supports_m1_validation",
            "documentation",
            contract_stage,
            sorted(ACCEPTED_CONTRACT_STAGES),
            contract_stage in ACCEPTED_CONTRACT_STAGES,
            notes=(
                "M1 accepts the raw-source contract before M2, or the promoted canonical M2 "
                "contract after Silver finalization."
            ),
        ),
    ]
    return checks, summarize_checks(checks)


def build_contract_from_dictionary(
    *,
    dictionary_path: Path = DOCS_DIR / "data_dictionary.csv",
    contract_path: Path = CONFIG_DIR / "data_contract.yaml",
) -> None:
    """Build an isolated M1 raw-source contract from a dictionary.

    This helper is retained for tests/tooling. It refuses to overwrite the shared
    canonical M2 contract at config/data_contract.yaml once that release exists.
    """
    canonical_contract = (CONFIG_DIR / "data_contract.yaml").resolve()
    target_contract = contract_path.resolve()
    if target_contract == canonical_contract and contract_path.exists():
        existing = yaml.safe_load(contract_path.read_text(encoding="utf-8")) or {}
        if str(existing.get("stage") or "").strip() == "milestone_2_canonical_release":
            raise RuntimeError(
                "Refusing to overwrite the canonical M2 Data Contract with an M1-only "
                "contract. Pass an explicit non-canonical contract_path for isolated M1 tooling."
            )

    rows = load_dictionary(dictionary_path)
    payload = {
        "contract_version": "m1-raw-contract-v1",
        "dataset": "external_raw_source",
        "stage": "milestone_1_source_contract",
        "expected_column_count": len(rows),
        "expected_row_count": None,
        "primary_key": None,
        "global_assumptions": {
            "price_unit": "toman_assumed_unconfirmed",
            "price_observation_type": "asking_price",
            "raw_is_read_only": True,
            "no_blanket_zero_imputation": True,
            "silver_published_after_m2_quality_gate": True,
        },
        "required_column_metadata": REQUIRED_FIELDS,
        "columns": rows,
    }
    atomic_write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False, width=120),
        contract_path,
    )


def write_validation(checks: list[Check], output_path: Path) -> None:
    atomic_write_csv(pl.DataFrame([check.__dict__ for check in checks]), output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate that the shared canonical Data Dictionary/Data Contract fully "
            "cover the Milestone 1 Raw schema."
        )
    )
    parser.add_argument("--raw", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checks, summary = validate_m1_documentation(args.raw)
    print("M1 DOCUMENTATION VALIDATION COMPLETED")
    for check in checks:
        print(
            f"{check.status}: {check.check_id} | "
            f"actual={check.actual} | expected={check.expected}"
        )
    print(f"overall_status: {summary['overall_status']}")
    if not summary["ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
