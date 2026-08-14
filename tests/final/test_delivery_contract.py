from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

from src.final_reporting.build_executive_summary import DEFAULT_OUTPUT as EXECUTIVE_OUTPUT
from src.final_reporting.build_technical_report import (
    CANONICAL_PRICE_REGIMES,
    DEFAULT_OUTPUT as TECHNICAL_OUTPUT,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_runner() -> tuple[Path, ModuleType]:
    # Transition-safe: current snapshot stores the runner under src/, while the
    # frozen architecture/README targets scripts/run_pipeline.py. Exactly one
    # orchestrator must exist; after the move this test remains valid.
    candidates = [
        PROJECT_ROOT / "scripts" / "run_pipeline.py",
        PROJECT_ROOT / "src" / "run_pipeline.py",
    ]
    existing = [path for path in candidates if path.exists()]
    assert len(existing) == 1, f"Expected exactly one canonical run_pipeline.py, found: {existing}"
    path = existing[0]
    spec = importlib.util.spec_from_file_location("ihmi_test_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return path, module


def test_single_pipeline_orchestrator_covers_m1_through_reports() -> None:
    _, runner = _load_runner()
    assert runner.GROUP_ORDER == ["m1", "m2", "m3", "m4", "reports"]
    assert runner._selected_groups("m2", "m4") == ["m2", "m3", "m4"]
    assert runner.MANIFEST_PATH.as_posix().endswith(
        "outputs/qa/final_pipeline/final_pipeline_manifest.json"
    )


def test_final_reports_are_exact_two_consolidated_professor_facing_outputs() -> None:
    assert EXECUTIVE_OUTPUT.as_posix() == "reports/final/Executive_Summary.md"
    assert TECHNICAL_OUTPUT.as_posix() == "reports/final/Technical_Report.md"
    assert len(CANONICAL_PRICE_REGIMES) == 9
    assert set(CANONICAL_PRICE_REGIMES) == {
        "sale",
        "rent_plus_deposit",
        "full_deposit",
        "rent_only",
        "rent_negotiable",
        "rent_unknown_or_incomplete",
        "temporary_rent",
        "service",
        "unknown",
    }


def test_final_notebook_and_dashboard_contract_surface_exist() -> None:
    assert (PROJECT_ROOT / "notebooks" / "final_analysis.ipynb").is_file()
    required_dashboard_contracts = {
        "dashboard_table_contract.csv",
        "dashboard_relationship_contract.csv",
        "dashboard_page_contract.csv",
        "dashboard_filter_contract.csv",
        "dashboard_visual_contract.csv",
        "dashboard_measure_catalog.csv",
        "measures.dax",
    }
    contract_dir = PROJECT_ROOT / "dashboard" / "contracts"
    observed = {path.name for path in contract_dir.iterdir() if path.is_file()}
    assert required_dashboard_contracts.issubset(observed)


def test_final_notebook_is_valid_json_even_before_runtime_acceptance() -> None:
    notebook = PROJECT_ROOT / "notebooks" / "final_analysis.ipynb"
    payload = json.loads(notebook.read_text(encoding="utf-8"))
    assert payload.get("nbformat") == 4
    assert isinstance(payload.get("cells"), list)
    assert payload["cells"], "final_analysis.ipynb must contain at least one cell"
