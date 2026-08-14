from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.config import configured_path  # noqa: E402
from src.common.io_utils import atomic_write_json  # noqa: E402

ORCHESTRATOR_VERSION = "final-pipeline-orchestrator-v1.1-governance-environment"
GROUP_ORDER = ["m1", "m2", "m3", "m4", "reports"]
MANIFEST_PATH = PROJECT_ROOT / "outputs" / "qa" / "final_pipeline" / "final_pipeline_manifest.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def module_command(module: str, *args: str) -> list[str]:
    return [sys.executable, "-m", module, *args]


def _display_command(command: list[str]) -> str:
    # Keep the manifest portable: do not persist the absolute interpreter path.
    values = ["python" if index == 0 else value for index, value in enumerate(command)]
    return " ".join(values)


def _run_command(
    command: list[str],
    *,
    stage_id: str,
    group: str,
    dry_run: bool,
    records: list[dict[str, Any]],
) -> None:
    rendered = _display_command(command)
    print(f"\n[{group.upper()}] {stage_id}\n> {rendered}")
    if dry_run:
        records.append(
            {
                "stage_id": stage_id,
                "group": group,
                "command": rendered,
                "status": "DRY_RUN",
                "elapsed_seconds": 0.0,
            }
        )
        return

    started = time.perf_counter()
    env = os.environ.copy()
    env["IHMI_PROJECT_ROOT"] = str(PROJECT_ROOT)
    try:
        subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)
    except subprocess.CalledProcessError as exc:
        records.append(
            {
                "stage_id": stage_id,
                "group": group,
                "command": rendered,
                "status": "FAIL",
                "return_code": exc.returncode,
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            }
        )
        raise
    records.append(
        {
            "stage_id": stage_id,
            "group": group,
            "command": rendered,
            "status": "PASS",
            "return_code": 0,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }
    )



def _spatial_bonus_args() -> tuple[str, ...]:
    """Use the project's frozen external spatial references when available.

    Reverse geocoding prefers a reviewed sanitized replay sample, then an internal
    cache-only fallback, so Restart/Run-All does not depend on live network access.
    """
    args: list[str] = []
    boundary = PROJECT_ROOT / "external_data" / "reference" / "geoboundaries_irn_adm2.geojson"
    metadata = PROJECT_ROOT / "external_data" / "reference" / "geoboundaries_irn_adm2_metadata.json"
    qa_dir = PROJECT_ROOT / "outputs" / "qa" / "milestone_3" / "spatial_quality"
    replay = qa_dir / "reverse_geocode_validation_sample.csv"
    cache = qa_dir / "reverse_geocode_cache.json"
    if boundary.exists():
        args += ["--boundary-geojson", str(boundary), "--boundary-sample-size", "5000"]
        if metadata.exists():
            args += ["--boundary-metadata-json", str(metadata)]
    if replay.exists():
        args += [
            "--reverse-geocode-sample-size", "80",
            "--reverse-geocode-replay-sample", str(replay),
        ]
    elif cache.exists():
        args += ["--reverse-geocode-sample-size", "80", "--reverse-geocode-cache-only"]
    return tuple(args)

def _assert_text_validation_ready(dry_run: bool, records: list[dict[str, Any]]) -> None:
    stage_id = "m3_text_manual_validation_gate"
    path = PROJECT_ROOT / "outputs" / "qa" / "milestone_3" / "text_analysis" / "keyword_precision_summary.csv"
    if dry_run:
        print(f"\n[M3] {stage_id}\n> validate {path.relative_to(PROJECT_ROOT).as_posix()}")
        records.append(
            {
                "stage_id": stage_id,
                "group": "m3",
                "command": "validate outputs/qa/milestone_3/text_analysis/keyword_precision_summary.csv",
                "status": "DRY_RUN",
                "elapsed_seconds": 0.0,
            }
        )
        return

    if not path.exists():
        raise FileNotFoundError(
            "Manual text-validation summary is missing. Run text_features, label "
            "keyword_manual_validation.csv, then rerun the pipeline."
        )
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "validation_status" not in rows[0]:
        raise RuntimeError("keyword_precision_summary.csv does not expose validation_status.")
    not_ready = [
        str(row.get("keyword", "unknown"))
        for row in rows
        if str(row.get("validation_status", "")).strip().upper() != "PASS"
    ]
    if not_ready:
        raise RuntimeError(
            "Manual text precision validation is not PASS for: " + ", ".join(not_ready) + ". "
            "Complete keyword_manual_validation.csv before the controlled text-price stage."
        )
    records.append(
        {
            "stage_id": stage_id,
            "group": "m3",
            "command": "validate outputs/qa/milestone_3/text_analysis/keyword_precision_summary.csv",
            "status": "PASS",
            "elapsed_seconds": 0.0,
        }
    )


def _selected_groups(start_at: str, stop_after: str) -> list[str]:
    start = GROUP_ORDER.index(start_at)
    stop = GROUP_ORDER.index(stop_after)
    if start > stop:
        raise ValueError("--start-at must not come after --stop-after.")
    return GROUP_ORDER[start : stop + 1]


def run_pipeline(*, start_at: str, stop_after: str, rebuild_silver: bool, dry_run: bool) -> Path:
    selected = _selected_groups(start_at, stop_after)
    records: list[dict[str, Any]] = []
    started_at = utc_now()
    status = "PASS"
    error: str | None = None

    try:
        if "m1" in selected:
            for stage_id, module in [
                ("m1_data_loading", "src.milestone_1.data_loading.data_loading"),
                ("m1_documentation", "src.milestone_1.documentation.data_documentation"),
                ("m1_data_audit", "src.milestone_1.audit.data_audit"),
                ("m1_closeout", "src.milestone_1.milestone1_closeout"),
            ]:
                _run_command(module_command(module), stage_id=stage_id, group="m1", dry_run=dry_run, records=records)

        if "m2" in selected:
            silver = configured_path("silver_master")
            if dry_run or rebuild_silver or not silver.exists():
                args = ("--overwrite",) if rebuild_silver and silver.exists() else ()
                _run_command(
                    module_command("src.milestone_2.silver_master.build_silver_master", *args),
                    stage_id="m2_silver_master",
                    group="m2",
                    dry_run=dry_run,
                    records=records,
                )
            else:
                print(
                    "\n[M2] m2_silver_master\n> reuse existing reviewed data/silver/silver_master.parquet "
                    "(use --rebuild-silver for a clean rebuild)"
                )
                records.append(
                    {
                        "stage_id": "m2_silver_master",
                        "group": "m2",
                        "command": "reuse existing reviewed data/silver/silver_master.parquet",
                        "status": "REUSED",
                        "elapsed_seconds": 0.0,
                    }
                )
            _run_command(
                module_command("src.milestone_2.milestone2_closeout"),
                stage_id="m2_closeout",
                group="m2",
                dry_run=dry_run,
                records=records,
            )

        if "m3" in selected:
            m3_stages = [
                ("m3_analysis_populations", "src.milestone_3.analysis_populations.analysis_populations", ()),
                ("m3_spatial_quality", "src.milestone_3.spatial_quality.spatial_quality", _spatial_bonus_args()),
                ("m3_market_map", "src.milestone_3.market_map.market_map", ()),
                ("m3_monthly_market", "src.milestone_3.monthly_market.monthly_market_summary", ()),
                ("m3_market_temperature", "src.milestone_3.market_temperature.market_temperature", ()),
                ("m3_analysis_ready", "src.milestone_3.price_drivers.analysis_ready_features", ()),
                ("m3_price_drivers", "src.milestone_3.price_drivers.price_drivers", ()),
                ("m3_price_driver_figures", "src.milestone_3.price_drivers.price_driver_figures", ()),
                ("m3_seller_type", "src.milestone_3.seller_type_comparison.seller_type_comparison", ()),
                ("m3_text_features", "src.milestone_3.text_analysis.text_features", ()),
            ]
            for stage_id, module, stage_args in m3_stages:
                _run_command(module_command(module, *stage_args), stage_id=stage_id, group="m3", dry_run=dry_run, records=records)
            _assert_text_validation_ready(dry_run, records)
            for stage_id, module in [
                ("m3_text_price_signals", "src.milestone_3.text_analysis.text_price_signals"),
                ("m3_market_segmentation", "src.milestone_3.market_segmentation.market_segmentation"),
                ("m3_closeout", "src.milestone_3.milestone3_closeout"),
            ]:
                _run_command(module_command(module), stage_id=stage_id, group="m3", dry_run=dry_run, records=records)

        if "m4" in selected:
            _run_command(
                module_command(
                    "src.milestone_4.gold.run_m4_gold",
                    "--reset-gold-structure",
                    "--reset-dashboard-metadata",
                ),
                stage_id="m4_gold_build_and_qa",
                group="m4",
                dry_run=dry_run,
                records=records,
            )

        if "reports" in selected:
            for stage_id, module in [
                ("report_environment_snapshot", "src.final_reporting.environment_snapshot"),
                ("report_statistical_governance", "src.final_reporting.statistical_governance"),
                ("report_executive_summary", "src.final_reporting.build_executive_summary"),
                ("report_technical_report", "src.final_reporting.build_technical_report"),
            ]:
                _run_command(module_command(module), stage_id=stage_id, group="reports", dry_run=dry_run, records=records)

    except Exception as exc:
        status = "FAIL"
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        payload: dict[str, Any] = {
            "version": ORCHESTRATOR_VERSION,
            "started_at_utc": started_at,
            "ended_at_utc": utc_now(),
            "status": status if not dry_run else "DRY_RUN",
            "start_at": start_at,
            "stop_after": stop_after,
            "rebuild_silver": rebuild_silver,
            "dry_run": dry_run,
            "stages": records,
            "error": error,
            "contracts": {
                "orchestration_only": True,
                "business_logic_duplicated_in_runner": False,
                "final_notebook_may_call_runner": True,
                "api_not_run_here": True,
            },
            "expected_final_outputs": {
                "gold_qa": "data/gold/qa/gold_qa_manifest.json",
                "environment_versions": "outputs/qa/final_pipeline/environment_versions.json",
                "statistical_governance_manifest": "outputs/qa/final_reporting/statistical_governance_manifest.json",
                "statistical_governance_checks": "outputs/qa/final_reporting/statistical_governance_checks.csv",
                "executive_summary": "reports/final/Executive_Summary.md",
                "technical_report": "reports/final/Technical_Report.md",
            },
        }
        atomic_write_json(payload, MANIFEST_PATH)

    return MANIFEST_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Orchestrate the canonical IHMI pipeline without duplicating analytical logic. "
            "Designed for reproducible project runs and final_analysis.ipynb."
        )
    )
    parser.add_argument("--start-at", choices=GROUP_ORDER, default="m1")
    parser.add_argument("--stop-after", choices=GROUP_ORDER, default="reports")
    parser.add_argument(
        "--rebuild-silver",
        action="store_true",
        help="Rebuild Silver from Raw; adds --overwrite only when an existing Silver Master is present.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the planned canonical commands without executing them.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_pipeline(
        start_at=args.start_at,
        stop_after=args.stop_after,
        rebuild_silver=args.rebuild_silver,
        dry_run=args.dry_run,
    )
    print(f"\nFINAL PIPELINE ORCHESTRATOR COMPLETED\nmanifest: {manifest.relative_to(PROJECT_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
