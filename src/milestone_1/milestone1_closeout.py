from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl
import yaml

from src.common.io_utils import atomic_write_csv, atomic_write_json
from src.common.paths import CONFIG_DIR, DOCS_DIR, OUTPUTS_DIR, relative_to_project
from src.common.validation import Check, make_check, summarize_checks


AGREED_M1_OUTPUTS = {
    "data_contract": CONFIG_DIR / "data_contract.yaml",
    "data_dictionary": DOCS_DIR / "data_dictionary.csv",
    "data_audit_report": OUTPUTS_DIR / "tables" / "milestone_1" / "data_audit_report.csv",
}


def _nonempty_file(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def run_closeout() -> dict[str, object]:
    """Check only the agreed Milestone 1 deliverables and publish closeout status."""
    checks: list[Check] = []

    for name, path in AGREED_M1_OUTPUTS.items():
        checks.append(
            make_check(
                f"{name}_exists_nonempty",
                "agreed_output",
                _nonempty_file(path),
                True,
                _nonempty_file(path),
                notes=relative_to_project(path),
            )
        )

    # Minimal format checks only; no new analysis is performed here.
    dictionary_path = AGREED_M1_OUTPUTS["data_dictionary"]
    if _nonempty_file(dictionary_path):
        dictionary = pl.read_csv(dictionary_path, infer_schema_length=10000)
        required = {
            "column_name", "business_meaning", "raw_dtype", "expected_dtype", "unit",
            "allowed_values", "missing_meaning", "cleaning_rule", "quality_risk", "owner",
        }
        checks.append(
            make_check(
                "data_dictionary_required_columns",
                "agreed_output",
                sorted(set(dictionary.columns) & required),
                sorted(required),
                required.issubset(set(dictionary.columns)),
            )
        )

    contract_path = AGREED_M1_OUTPUTS["data_contract"]
    if _nonempty_file(contract_path):
        contract = yaml.safe_load(contract_path.read_text(encoding="utf-8")) or {}
        checks.append(
            make_check(
                "data_contract_has_columns",
                "agreed_output",
                len(contract.get("columns", []) or []),
                ">0",
                bool(contract.get("columns", []) or []),
            )
        )

    audit_path = AGREED_M1_OUTPUTS["data_audit_report"]
    if _nonempty_file(audit_path):
        audit = pl.read_csv(audit_path, infer_schema_length=10000)
        checks.append(
            make_check(
                "data_audit_report_nonempty",
                "agreed_output",
                audit.height,
                ">0",
                audit.height > 0,
            )
        )

    summary = summarize_checks(checks)
    qa_dir = OUTPUTS_DIR / "qa" / "milestone_1"
    qa_dir.mkdir(parents=True, exist_ok=True)
    validation_path = qa_dir / "milestone1_validation.csv"
    summary_path = qa_dir / "milestone1_summary.json"

    atomic_write_csv(pl.DataFrame([c.__dict__ for c in checks]), validation_path)
    atomic_write_json(
        {
            **summary,
            "milestone": 1,
            "m2_entry_ready": bool(summary["ready"]),
            "checked_outputs": {name: relative_to_project(path) for name, path in AGREED_M1_OUTPUTS.items()},
        },
        summary_path,
    )
    return {**summary, "validation_path": validation_path, "summary_path": summary_path}


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(description="Close Milestone 1 by checking only the agreed M1 outputs.").parse_args()


def main() -> None:
    parse_args()
    result = run_closeout()
    print("MILESTONE 1 CLOSEOUT COMPLETED")
    print(f"overall_status: {result['overall_status']}")
    print(f"m2_entry_ready: {result['ready']}")
    print(f"validation: {relative_to_project(result['validation_path'])}")
    print(f"summary: {relative_to_project(result['summary_path'])}")
    if not result["ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
