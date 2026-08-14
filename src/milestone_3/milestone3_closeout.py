from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from src.common.io_utils import atomic_write_csv, atomic_write_json
from src.common.paths import OUTPUTS_DIR, relative_to_project

EXPECTED_OUTPUTS: dict[str, Path] = {
    "analysis_population_summary": OUTPUTS_DIR / "tables" / "milestone_3" / "analysis_populations" / "analysis_population_summary.csv",
    "spatial_quality_summary": OUTPUTS_DIR / "tables" / "milestone_3" / "spatial_quality" / "spatial_quality_summary.csv",
    "city_market_summary": OUTPUTS_DIR / "tables" / "milestone_3" / "market_map" / "city_market_summary.csv",
    "neighborhood_market_summary": OUTPUTS_DIR / "tables" / "milestone_3" / "market_map" / "neighborhood_market_summary.csv",
    "monthly_market_summary": OUTPUTS_DIR / "tables" / "milestone_3" / "monthly_market" / "monthly_market_summary.csv",
    "market_temperature_summary": OUTPUTS_DIR / "tables" / "milestone_3" / "market_temperature" / "market_temperature_summary.csv",
    "price_driver_summary": OUTPUTS_DIR / "tables" / "milestone_3" / "price_drivers" / "price_driver_summary.csv",
    "seller_type_comparison_summary": OUTPUTS_DIR / "tables" / "milestone_3" / "seller_type_comparison" / "seller_type_comparison_summary.csv",
    "text_signal_summary": OUTPUTS_DIR / "tables" / "milestone_3" / "text_analysis" / "text_signal_summary.csv",
    "segment_profile": OUTPUTS_DIR / "tables" / "milestone_3" / "market_segmentation" / "segment_profile.csv",
    "listing_segments": OUTPUTS_DIR / "model_artifacts" / "milestone_3" / "market_segmentation" / "listing_segments.parquet",
}

CHECK_FILES: dict[str, Path] = {
    "analysis_populations": OUTPUTS_DIR / "qa" / "milestone_3" / "analysis_populations" / "analysis_population_checks.csv",
    "spatial_quality": OUTPUTS_DIR / "qa" / "milestone_3" / "spatial_quality" / "spatial_quality_checks.csv",
    "market_map": OUTPUTS_DIR / "qa" / "milestone_3" / "market_map" / "market_map_checks.csv",
    "monthly_market": OUTPUTS_DIR / "qa" / "milestone_3" / "monthly_market" / "monthly_market_checks.csv",
    "market_temperature": OUTPUTS_DIR / "qa" / "milestone_3" / "market_temperature" / "market_temperature_checks.csv",
    "price_drivers": OUTPUTS_DIR / "qa" / "milestone_3" / "price_drivers" / "price_driver_checks.csv",
    "seller_type_comparison": OUTPUTS_DIR / "qa" / "milestone_3" / "seller_type_comparison" / "seller_type_checks.csv",
    "text_analysis": OUTPUTS_DIR / "qa" / "milestone_3" / "text_analysis" / "text_signal_checks.csv",
    "market_segmentation": OUTPUTS_DIR / "qa" / "milestone_3" / "market_segmentation" / "segmentation_checks.csv",
}


def closeout() -> tuple[pl.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    missing_outputs: list[str] = []
    for name, path in EXPECTED_OUTPUTS.items():
        exists = path.exists()
        if not exists:
            missing_outputs.append(name)
        rows.append({
            "check": f"output:{name}", "actual": "present" if exists else "missing",
            "expected": "present", "status": "PASS" if exists else "FAIL", "critical": True,
            "notes": relative_to_project(path),
        })

    for task, path in CHECK_FILES.items():
        if not path.exists():
            rows.append({"check": f"task:{task}", "actual": "checks_missing", "expected": "no critical FAIL", "status": "FAIL", "critical": True, "notes": relative_to_project(path)})
            continue
        frame = pl.read_csv(path)
        critical_failures = frame.filter((pl.col("status") == "FAIL") & pl.col("critical").cast(pl.Boolean)).height
        reviews = frame.filter(pl.col("status") == "REVIEW").height
        status = "FAIL" if critical_failures else ("REVIEW" if reviews else "PASS")
        rows.append({
            "check": f"task:{task}", "actual": f"critical_failures={critical_failures}; reviews={reviews}",
            "expected": "critical_failures=0", "status": status, "critical": True,
            "notes": relative_to_project(path),
        })

    validation = pl.DataFrame(rows)
    critical_failures = validation.filter((pl.col("status") == "FAIL") & pl.col("critical")).height
    review_count = validation.filter(pl.col("status") == "REVIEW").height
    overall = "FAIL" if critical_failures else ("REVIEW" if review_count else "PASS")
    summary = {
        "overall_status": overall,
        "ready": critical_failures == 0,
        "critical_failures": critical_failures,
        "review_count": review_count,
        "check_count": validation.height,
        "milestone": 3,
        "section_28_entry_ready": critical_failures == 0,
        "checked_outputs": {name: relative_to_project(path) for name, path in EXPECTED_OUTPUTS.items()},
        "missing_outputs": missing_outputs,
    }
    return validation, summary


def main() -> None:
    validation, summary = closeout()
    qa_dir = OUTPUTS_DIR / "qa" / "milestone_3"
    qa_dir.mkdir(parents=True, exist_ok=True)
    validation_path = qa_dir / "milestone3_validation.csv"
    summary_path = qa_dir / "milestone3_summary.json"
    atomic_write_csv(validation, validation_path)
    atomic_write_json(summary, summary_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
