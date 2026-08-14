from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from src.common.config import configured_path
from src.common.io_utils import atomic_write_json
from src.common.paths import OUTPUTS_DIR, relative_to_project


REQUIRED_OUTPUTS = {
    "silver_master": lambda: configured_path("silver_master"),
    "data_quality_action_table": lambda: OUTPUTS_DIR / "tables" / "milestone_2" / "data_quality_action_table.csv",
    "standardization_summary": lambda: OUTPUTS_DIR / "tables" / "milestone_2" / "standardization" / "standardization_summary.csv",
    "missingness_action_table": lambda: OUTPUTS_DIR / "tables" / "milestone_2" / "missingness" / "missingness_action_table.csv",
    "duplicate_summary": lambda: OUTPUTS_DIR / "tables" / "milestone_2" / "duplicates" / "duplicate_summary.csv",
    "duplicate_supply_impact": lambda: OUTPUTS_DIR / "tables" / "milestone_2" / "duplicates" / "duplicate_supply_impact.csv",
    "currency_validation_summary": lambda: OUTPUTS_DIR / "tables" / "milestone_2" / "currency" / "currency_validation_summary.csv",
    "currency_inference_interpretation": lambda: OUTPUTS_DIR / "tables" / "milestone_2" / "currency" / "currency_inference_interpretation.md",
    "price_regime_review_summary": lambda: OUTPUTS_DIR / "tables" / "milestone_2" / "price_regimes" / "price_regime_review_summary.csv",
    "outlier_summary": lambda: OUTPUTS_DIR / "tables" / "milestone_2" / "outliers" / "outlier_summary.csv",
    "outlier_sensitivity": lambda: OUTPUTS_DIR / "tables" / "milestone_2" / "outliers" / "outlier_sensitivity.csv",
    "final_metric_summary": lambda: OUTPUTS_DIR / "tables" / "milestone_2" / "final_metrics" / "final_metric_summary.csv",
    "quality_gate_summary": lambda: OUTPUTS_DIR / "tables" / "milestone_2" / "quality_gate" / "quality_gate_summary.csv",
}


def _quality_gate_status(path: Path) -> tuple[int, int]:
    """Return critical failures and review items from the canonical M2 quality gate."""
    frame = pl.read_csv(path, infer_schema_length=1000)
    if "status" not in frame.columns:
        return 1, 0

    status = pl.col("status").cast(pl.String).str.to_uppercase()
    if "critical" in frame.columns:
        critical = pl.col("critical").cast(pl.Boolean, strict=False).fill_null(False)
    else:
        critical = pl.lit(True)

    critical_failures = frame.filter((status == "FAIL") & critical).height
    review_items = frame.filter(
        (status == "REVIEW") | ((status == "FAIL") & ~critical)
    ).height
    return critical_failures, review_items


def main() -> None:
    checked = {name: path_fn() for name, path_fn in REQUIRED_OUTPUTS.items()}
    missing = [name for name, path in checked.items() if not path.exists()]

    critical_failures = 0
    review_count = 0
    gate_path = checked["quality_gate_summary"]
    if gate_path.exists():
        critical_failures, review_count = _quality_gate_status(gate_path)

    ready = not missing and critical_failures == 0
    overall = "FAIL" if not ready else ("REVIEW" if review_count else "PASS")

    payload = {
        "overall_status": overall,
        "ready": ready,
        "critical_failures": critical_failures + len(missing),
        "review_count": review_count,
        "check_count": len(checked),
        "milestone": 2,
        "m3_entry_ready": ready,
        "checked_outputs": {
            name: relative_to_project(path) for name, path in checked.items()
        },
        "missing_outputs": missing,
    }

    output = OUTPUTS_DIR / "qa" / "milestone_2" / "milestone2_summary.json"
    atomic_write_json(payload, output)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    raise SystemExit(0 if ready else 1)


if __name__ == "__main__":
    main()
