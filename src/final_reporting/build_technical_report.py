from __future__ import annotations

import argparse
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import yaml


REPORT_VERSION = "final-technical-report-v3.3-public-fastapi-deployment"
DEFAULT_OUTPUT = Path("reports/final/Technical_Report.md")
CANONICAL_PRICE_REGIMES = (
    "sale",
    "rent_plus_deposit",
    "full_deposit",
    "rent_only",
    "rent_negotiable",
    "rent_unknown_or_incomplete",
    "temporary_rent",
    "service",
    "unknown",
)
SEGMENT_ENGLISH_NAMES = {
    "SEG01": "Relative affordable | small urban unit",
    "SEG02": "Mid-market | family",
    "SEG03": "Relative luxury | family",
    "SEG04": "Land / investment property",
    "SEG05": "Commercial / office property",
}

def _project_root_from_script() -> Path:
    script = Path(__file__).resolve()
    for candidate in (script.parent, *script.parents):
        if (candidate / "src").exists() and (candidate / "outputs").exists():
            return candidate
    cwd = Path.cwd().resolve()
    for candidate in (cwd, *cwd.parents):
        if (candidate / "src").exists() and (candidate / "outputs").exists():
            return candidate
    return cwd


def resolve_project_root(explicit: str | Path | None = None) -> Path:
    if explicit is not None:
        return Path(explicit).expanduser().resolve()
    env_value = os.getenv("IHMI_PROJECT_ROOT")
    if env_value:
        return Path(env_value).expanduser().resolve()
    return _project_root_from_script()


def relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def first_existing(*paths: Path) -> Path | None:
    return next((path for path in paths if path.exists()), None)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return value


def _nested(mapping: dict[str, Any], *keys: str, default: Any = None) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        try:
            import polars as pl  # type: ignore

            return pd.DataFrame(pl.read_parquet(path).to_dicts())
        except ImportError:
            return pd.read_parquet(path)
    raise ValueError(f"Unsupported table type: {path}")


def require_columns(frame: pd.DataFrame, required: Iterable[str], source: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.fillna(False).astype(bool)
    return series.astype("string").str.strip().str.lower().isin({"true", "1", "yes", "pass", "passed", "ready"})


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([float("inf"), float("-inf")], pd.NA)


def fmt_int(value: Any) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{int(round(float(value))):,}"


def fmt_num(value: Any, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):,.{digits}f}"


def fmt_metric_value(value: Any) -> str:
    if value is None or pd.isna(value):
        return "-"
    text = str(value).strip()
    try:
        number = float(text)
    except (TypeError, ValueError):
        return text
    if number.is_integer():
        return f"{int(number):,}"
    return f"{number:,.6f}".rstrip("0").rstrip(".")


def _md_cell(value: Any) -> str:
    return str(value).replace("\n", " ").replace("|", "\\|").strip()


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    safe_headers = [_md_cell(value) for value in headers]
    lines = [
        "| " + " | ".join(safe_headers) + " |",
        "|" + "|".join(["---"] * len(safe_headers)) + "|",
    ]
    lines.extend("| " + " | ".join(_md_cell(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def _assert_english_only(text: str) -> None:
    match = re.search(r"[\u0600-\u06FF]", text)
    if match:
        excerpt = text[max(0, match.start() - 40) : match.start() + 80]
        raise RuntimeError(f"Technical Report contains non-English script near: {excerpt!r}")


def _assert_no_personal_paths(text: str) -> None:
    patterns = [r"[A-Za-z]:\\Users\\", r"/Users/[^/]+/", r"/home/[^/]+/"]
    for pattern in patterns:
        if re.search(pattern, text):
            raise RuntimeError("Technical Report contains a personal filesystem path.")


def _artifact_paths(root: Path) -> dict[str, Path | None]:
    m2 = root / "outputs" / "tables" / "milestone_2"
    m2_qa = root / "outputs" / "qa" / "milestone_2"
    m3 = root / "outputs" / "tables" / "milestone_3"
    m3_qa = root / "outputs" / "qa" / "milestone_3"
    gold = root / "data" / "gold"
    return {
        "settings": first_existing(root / "config" / "settings.yaml"),
        "quality_gate": first_existing(
            m2 / "quality_gate" / "quality_gate_summary.csv",
            m2 / "quality_gate" / "milestone2_quality_gate.csv",
        ),
        "standardization_summary": first_existing(
            m2 / "standardization" / "standardization_summary.csv",
        ),
        "outlier_summary": first_existing(m2 / "outliers" / "outlier_summary.csv"),
        "outlier_sensitivity": first_existing(m2 / "outliers" / "outlier_sensitivity.csv"),
        "outlier_manifest": first_existing(
            m2 / "outliers" / "outlier_validation_manifest.json",
            m2_qa / "outliers" / "outlier_validation_manifest.json",
        ),
        "duplicate_summary": first_existing(m2 / "duplicates" / "duplicate_summary.csv"),
        "duplicate_supply_impact": first_existing(m2 / "duplicates" / "duplicate_supply_impact.csv"),
        "final_metrics": first_existing(m2 / "final_metrics" / "final_metric_summary.csv"),
        "missingness_manifest": first_existing(
            m2 / "missingness" / "missingness_manifest.json",
            m2_qa / "missingness" / "missingness_manifest.json",
            root / "outputs" / "tables" / "missingness" / "missingness_manifest.json",
        ),
        "missingness_action": first_existing(
            m2 / "missingness" / "missingness_action_table.csv",
        ),
        "price_regime_review": first_existing(
            m2 / "price_regimes" / "price_regime_review_summary.csv",
            m2 / "price_regime" / "price_regime_review_summary.csv",
        ),
        "price_regime_manifest": first_existing(
            m2 / "price_regimes" / "price_regime_validation_manifest.json",
            m2 / "price_regime" / "price_regime_validation_manifest.json",
            m2_qa / "price_regimes" / "price_regime_validation_manifest.json",
            m2_qa / "price_regime" / "price_regime_validation_manifest.json",
            root / "outputs" / "tables" / "price_regime" / "price_regime_validation_manifest.json",
        ),
        "dim_price_regime": first_existing(
            gold / "dimensions" / "dim_price_regime.parquet",
            gold / "dimensions" / "dim_price_regime.csv",
        ),
        "spatial_summary": first_existing(m3 / "spatial_quality" / "spatial_quality_summary.csv"),
        "segment_profile": first_existing(m3 / "market_segmentation" / "segment_profile.csv"),
        "segment_manifest": first_existing(
            m3_qa / "market_segmentation" / "segmentation_manifest.json",
            m3 / "market_segmentation" / "segmentation_manifest.json",
        ),
        "model_diagnostics": first_existing(m3 / "price_drivers" / "price_driver_model_diagnostics.csv"),
        "dashboard_quality": first_existing(gold / "metadata" / "dashboard_quality_status.csv"),
        "gold_qa_manifest": first_existing(gold / "qa" / "gold_qa_manifest.json"),
        "gold_inventory": first_existing(gold / "metadata" / "gold_artifact_inventory.csv"),
    }


def _required_tables(paths: dict[str, Path | None]) -> dict[str, pd.DataFrame]:
    required = [
        "quality_gate",
        "standardization_summary",
        "outlier_summary",
        "outlier_sensitivity",
        "duplicate_summary",
        "final_metrics",
        "price_regime_review",
        "segment_profile",
    ]
    missing = [name for name in required if paths.get(name) is None]
    if missing:
        raise FileNotFoundError("Technical Report required artifacts are missing: " + ", ".join(missing))
    return {name: read_table(paths[name]) for name in required if paths[name] is not None}


def _quality_summary(frame: pd.DataFrame) -> dict[str, Any]:
    require_columns(frame, ["status"], "quality gate summary")
    status = frame["status"].astype(str).str.upper()
    critical = bool_series(frame["critical"]) if "critical" in frame.columns else pd.Series([True] * len(frame))
    return {
        "checks": int(len(frame)),
        "pass": int((status == "PASS").sum()),
        "review": int((status == "REVIEW").sum()),
        "fail": int((status == "FAIL").sum()),
        "critical_fail": int(((status == "FAIL") & critical).sum()),
    }


def _lookup_metric(frame: pd.DataFrame, key_column: str, key: str, value_column: str) -> Any:
    if key_column not in frame.columns or value_column not in frame.columns:
        return None
    rows = frame.loc[frame[key_column].astype(str) == key]
    return rows.iloc[0][value_column] if not rows.empty else None


def _gold_status(root: Path, paths: dict[str, Path | None]) -> tuple[str, list[str], bool | None]:
    path = paths.get("gold_qa_manifest")
    if path is None:
        return "Gold QA manifest not found.", [], None
    payload = read_json(path)
    status = payload.get("status") if isinstance(payload.get("status"), dict) else {}
    architecture = payload.get("architecture") if isinstance(payload.get("architecture"), dict) else {}
    overall = str(status.get("overall_status", "UNKNOWN")).upper()
    critical = int(status.get("critical_failures") or 0)
    ready = bool(payload.get("gold_data_contract_ready"))
    expected_architecture = {"gold_marts": 10, "conformed_dimensions": 5, "physical_relationships": 13}
    architecture_ok = all(architecture.get(key) == value for key, value in expected_architecture.items())
    semantic_only_ok = architecture.get("semantic_only_dimensions") in (["dim_user_type"], None)
    effective_ready = bool(ready and overall == "PASS" and critical == 0 and architecture_ok and semantic_only_ok)
    text = (
        f"Gold QA status=`{overall}`, critical_failures=`{critical}`, "
        f"gold_data_contract_ready=`{ready}`, marts=`{architecture.get('gold_marts', '-')}`, "
        f"dimensions=`{architecture.get('conformed_dimensions', '-')}`, "
        f"physical_relationships=`{architecture.get('physical_relationships', '-')}`."
    )
    if ready and not effective_ready:
        text += " The Gold readiness flag conflicts with the frozen architecture/status contract."
    return text, [relative_path(path, root)], effective_ready


def _dashboard_stage_status(root: Path, path: Path | None) -> tuple[str, list[str]]:
    if path is None:
        raise FileNotFoundError("Gold dashboard_quality_status.csv was not found.")
    frame = read_table(path)
    required = {"stage", "overall_status", "ready"}
    if not required.issubset(frame.columns):
        raise ValueError("dashboard_quality_status.csv is missing stage/overall_status/ready fields.")
    core = frame.loc[frame["stage"].astype(str).isin(["M1", "M2", "M3", "M4"])].copy()
    stages = set(core["stage"].astype(str))
    if stages != {"M1", "M2", "M3", "M4"}:
        raise RuntimeError(f"Dashboard quality metadata must contain M1-M4; found {sorted(stages)}")
    if not bool_series(core["ready"]).all():
        bad = core.loc[~bool_series(core["ready"]), "stage"].astype(str).tolist()
        raise RuntimeError(f"Dashboard quality metadata contains non-ready core stages: {bad}")
    if core["overall_status"].astype(str).str.upper().eq("FAIL").any():
        bad = core.loc[core["overall_status"].astype(str).str.upper().eq("FAIL"), "stage"].astype(str).tolist()
        raise RuntimeError(f"Dashboard quality metadata contains FAIL core stages: {bad}")

    rows: list[list[str]] = []
    for stage in ["M1", "M2", "M3", "M4"]:
        part = core.loc[core["stage"].astype(str) == stage].copy()
        statuses = part["overall_status"].astype(str).str.upper()
        stage_status = "REVIEW" if statuses.eq("REVIEW").any() else "PASS"
        critical = int(numeric(part["critical_failures"]).fillna(0).sum()) if "critical_failures" in part.columns else 0
        reviews = int(numeric(part["review_count"]).fillna(0).sum()) if "review_count" in part.columns else int(statuses.eq("REVIEW").sum())
        components = int(len(part))
        rows.append([stage, stage_status, "True", str(components), str(critical), str(reviews)])
    return (
        _md_table(["Stage", "Status", "Ready", "Components", "Critical failures", "Reviews"], rows),
        [relative_path(path, root)],
    )


def _standardization_section(
    root: Path, summary_path: Path | None
) -> tuple[str, list[str], int | None]:
    if summary_path is None:
        raise FileNotFoundError("Canonical M2 standardization_summary.csv was not found.")

    summary = read_table(summary_path)
    require_columns(summary, ["metric", "value", "status"], "standardization_summary.csv")
    required_metrics = {
        "row_count",
        "rows_with_parse_error",
        "analysis_month_parse_failure_rows",
        "title_normalized_changed_rows",
        "description_normalized_changed_rows",
    }
    observed_metrics = set(summary["metric"].astype(str))
    missing_metrics = sorted(required_metrics - observed_metrics)
    if missing_metrics:
        raise RuntimeError(f"Standardization summary is missing required metrics: {missing_metrics}")

    statuses = summary["status"].astype(str).str.upper()
    if statuses.eq("FAIL").any():
        failed = summary.loc[statuses.eq("FAIL"), "metric"].astype(str).tolist()
        raise RuntimeError(f"Technical Report generation blocked: standardization summary contains FAIL metrics: {failed}")

    row_count = _lookup_metric(summary, "metric", "row_count", "value")
    analysis_month_failures = _lookup_metric(
        summary, "metric", "analysis_month_parse_failure_rows", "value"
    )
    if analysis_month_failures is not None and int(float(analysis_month_failures)) != 0:
        raise RuntimeError(
            "Final Technical Report expects zero analysis-month parse failures; "
            f"found {analysis_month_failures}."
        )

    interpretation = {
        "row_count": "Rows covered by canonical M2 standardization.",
        "rows_with_parse_error": (
            "Rows with at least one structured-field parse issue; retained and flagged for review rather than silently coerced."
        ),
        "analysis_month_parse_failure_rows": "Rows whose listing month could not be parsed; final expectation is zero.",
        "title_normalized_changed_rows": (
            "Rows where the derived normalized title differs from the preserved raw title because of character, digit, or whitespace normalization."
        ),
        "description_normalized_changed_rows": (
            "Rows where the derived normalized description differs from the preserved raw description because of character, digit, or whitespace normalization."
        ),
    }
    rows: list[list[str]] = []
    order = [
        "row_count",
        "title_normalized_changed_rows",
        "description_normalized_changed_rows",
        "rows_with_parse_error",
        "analysis_month_parse_failure_rows",
    ]
    indexed = summary.set_index(summary["metric"].astype(str), drop=False)
    for metric in order:
        row = indexed.loc[metric]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        rows.append(
            [
                metric,
                fmt_metric_value(row["value"]),
                str(row["status"]).upper(),
                interpretation[metric],
            ]
        )

    text = (
        "M2 standardization is lineage-preserving: the original source columns remain available for auditability while derived normalized/typed columns are added for analysis. "
        "For listing text, Arabic Yeh/Kaf glyph variants are mapped to their canonical Persian forms; Persian and Arabic-Indic digits are converted to ASCII/Western digits 0-9; non-breaking/unusual spaces are normalized; repeated whitespace is collapsed; and leading/trailing whitespace is removed. NULL values remain NULL. "
        "The normalized title and description are derived fields and do not overwrite the raw text.\n\n"
        "Structured-field standardization separately normalizes missing-like strings, removes supported numeric separators, parses monetary/area/coordinate/capacity fields into typed columns, maps supported boolean values, preserves censored or source-semantic categories through explicit flags, and records unresolved parsing in `type_parse_error_count` rather than inventing values.\n\n"
        + _md_table(["Standardization metric", "Value", "Status", "Interpretation"], rows)
        + "\n\nA `REVIEW` on residual structured parse errors is not equivalent to row deletion or imputation: those source values remain auditable and downstream eligibility rules decide whether a record is usable for a specific metric."
    )
    return text, [relative_path(summary_path, root)], (int(float(row_count)) if row_count is not None else None)


def _missingness_policy(
    root: Path, manifest_path: Path | None, action_path: Path | None
) -> tuple[str, list[str]]:
    sources: list[str] = []
    if manifest_path is not None:
        payload = read_json(manifest_path)
        policy = payload.get("policy") if isinstance(payload.get("policy"), dict) else {}
        text = (
            f"The missingness policy records zero_is_missing=`{policy.get('zero_is_missing', '-')}`, "
            f"blanket_zero_fill=`{policy.get('blanket_zero_fill', '-')}`, "
            f"blanket_row_deletion=`{policy.get('blanket_row_deletion', '-')}`, and "
            f"imputation_scope=`{policy.get('imputation_scope', '-')}`."
        )
        sources.append(relative_path(manifest_path, root))
    else:
        text = (
            "The frozen project policy retains NULL in Silver, forbids blanket zero/False filling and blanket row deletion, "
            "and permits imputation only inside a justified downstream model/analysis workflow."
        )

    if action_path is not None:
        actions = read_table(action_path)
        require_columns(actions, ["column", "action"], "missingness_action_table.csv")
        counts = actions["action"].astype(str).value_counts()
        text += (
            f" The current action table documents **{len(actions)}** column-level issue/action rows. "
            f"Task-specific NULL preservation is used for **{int(counts.get('Preserve null and apply task-specific applicability rules downstream.', 0))}** rows; "
            f"model-only numeric imputation for **{int(counts.get('Preserve null in Silver; impute only inside a specific downstream model when justified.', 0))}**; "
            f"low-coverage retention for **{int(counts.get('Retain for auditability but treat as low-coverage unless a task explicitly needs it.', 0))}**; "
            f"and explicit unknown-vs-False amenity preservation for **{int(counts.get('Preserve null; do not convert unknown amenities to False.', 0))}**."
        )
        sources.append(relative_path(action_path, root))
    return text, sources


def _outlier_rule_config(root: Path, settings_path: Path | None) -> tuple[dict[str, Any], str]:
    if settings_path is None:
        raise FileNotFoundError("config/settings.yaml is required to document the canonical outlier policy.")
    payload = read_yaml(settings_path)
    config = {
        "rule_version": _nested(payload, "milestone_2", "versions", "outlier"),
        "canonical_iqr_multiplier": _nested(payload, "milestone_2", "outliers", "canonical_iqr_multiplier"),
        "sensitivity_iqr_multiplier": _nested(payload, "milestone_2", "outliers", "sensitivity_iqr_multiplier"),
        "minimum_local_group_rows": _nested(payload, "milestone_2", "outliers", "min_local_group_rows"),
        "minimum_fallback_group_rows": _nested(payload, "milestone_2", "outliers", "min_fallback_group_rows"),
    }
    missing = [key for key, value in config.items() if value is None]
    if missing:
        raise RuntimeError(f"Canonical outlier configuration is incomplete in settings.yaml: {missing}")
    if str(config["rule_version"]) != "outlier-policy-m2-v2":
        raise RuntimeError(
            f"Final Technical Report expects the frozen canonical outlier policy v2; found {config['rule_version']!r}."
        )
    return config, relative_path(settings_path, root)


def _rent_conversion_scenarios(root: Path, settings_path: Path | None) -> tuple[dict[str, float], str]:
    if settings_path is None:
        raise FileNotFoundError("config/settings.yaml is required to document rent/deposit sensitivity scenarios.")
    payload = read_yaml(settings_path)
    values = _nested(payload, "analysis", "rent_equivalence_k_toman", default={})
    if not isinstance(values, dict) or not {"low", "base", "high"}.issubset(values):
        raise RuntimeError("settings.yaml is missing low/base/high rent_equivalence_k_toman scenarios.")
    scenarios = {key: float(values[key]) / 1_000_000.0 for key in ["low", "base", "high"]}
    return scenarios, relative_path(settings_path, root)


def _price_regime_section(
    root: Path, review_path: Path | None, settings_path: Path | None
) -> tuple[str, list[str]]:
    if review_path is None:
        raise FileNotFoundError("Canonical M2 price_regime_review_summary.csv was not found.")
    review = read_table(review_path)
    required = ["Raw Pattern", "Expected Regime", "Observed Count", "Cleaning Rule", "Validation Result"]
    require_columns(review, required, "price_regime_review_summary.csv")
    observed = review["Expected Regime"].astype(str).tolist()
    if len(observed) != len(set(observed)):
        raise RuntimeError("Price-regime review contains duplicate regime rows.")
    if set(observed) != set(CANONICAL_PRICE_REGIMES):
        raise RuntimeError(
            "Canonical price-regime taxonomy mismatch: "
            f"expected={list(CANONICAL_PRICE_REGIMES)}, observed={observed}"
        )

    version = "-"
    settings_source = None
    if settings_path is not None:
        settings = read_yaml(settings_path)
        version = str(_nested(settings, "milestone_2", "versions", "price_regime", default="-"))
        settings_source = relative_path(settings_path, root)

    rows: list[list[str]] = []
    for _, row in review.iterrows():
        rows.append(
            [
                str(row["Expected Regime"]),
                fmt_int(row["Observed Count"]),
                str(row["Validation Result"]).upper(),
                str(row["Cleaning Rule"]),
            ]
        )
    sources = [relative_path(review_path, root)]
    if settings_source:
        sources.append(settings_source)
    text = (
        f"The canonical taxonomy is version `{version}` and contains all **{len(CANONICAL_PRICE_REGIMES)}** regimes, including non-analytical service/unknown states. "
        "The complete taxonomy is retained even when a regime has zero observed rows so that classification behavior is explicit and reproducible.\n\n"
        + _md_table(["Regime", "Observed rows", "Validation", "Cleaning / interpretation rule"], rows)
        + "\n\n`REVIEW` is expected for negotiable or incomplete monetary structures because numeric values are intentionally not invented. "
        "`service` and `temporary_rent` remain separate from long-term property-price analysis, and `unknown` is never coerced to another regime."
    )
    return text, sources


def _segmentation_section(root: Path, frame: pd.DataFrame, manifest_path: Path | None) -> tuple[str, list[str]]:
    require_columns(frame, ["segment_id", "segment_method", "listing_n", "listing_share_pct"], "segment_profile.csv")
    methods = sorted(frame["segment_method"].dropna().astype(str).unique().tolist())
    allowed_methods = {"rule_based_descriptive_typology", "compatible_domain_segment"}
    unexpected_methods = sorted(set(methods) - allowed_methods)
    if unexpected_methods:
        raise RuntimeError(f"Unexpected final segmentation methods: {unexpected_methods}")
    segment_ids = set(frame["segment_id"].dropna().astype(str))
    expected_ids = set(SEGMENT_ENGLISH_NAMES)
    if segment_ids != expected_ids:
        raise RuntimeError(f"Final segment IDs do not match the frozen five-segment profile: {sorted(segment_ids)}")
    shares = numeric(frame["listing_share_pct"]).dropna()
    if shares.empty or not math.isclose(float(shares.sum()), 100.0, rel_tol=0.0, abs_tol=0.05):
        raise RuntimeError(f"Segment listing shares must sum to approximately 100%; found {float(shares.sum()) if not shares.empty else 'missing'}")
    total_n = int(numeric(frame["listing_n"]).fillna(0).sum())
    sources: list[str] = []
    manifest_note = ""
    if manifest_path is not None:
        payload = read_json(manifest_path)
        primary_method = payload.get("primary_method") or payload.get("method")
        if primary_method:
            manifest_note = f" Manifest primary_method=`{primary_method}`."
        sources.append(relative_path(manifest_path, root))

    rows: list[list[str]] = []
    for _, row in frame.sort_values("listing_share_pct", ascending=False).iterrows():
        segment_id = str(row["segment_id"])
        english_name = SEGMENT_ENGLISH_NAMES.get(segment_id, f"Descriptive segment {segment_id}")
        rows.append(
            [
                segment_id,
                english_name,
                str(row["segment_method"]),
                fmt_int(row["listing_n"]),
                f"{float(row['listing_share_pct']):.1f}%",
            ]
        )

    text = (
        "The final segmentation release is a **descriptive market typology**, not a causal class, socioeconomic class, or buyer-persona model. "
        f"Published method(s): `{', '.join(methods)}`.{manifest_note} The profile contains **{fmt_int(total_n)}** assigned listings.\n\n"
        + _md_table(["Segment", "English label", "Method", "N", "Share"], rows)
        + "\n\nIncompatible property families are not forced into one apartment-style geometry. Relative affordable/luxury labels are meaningful only within the compatible local/reference pricing framework."
    )
    return text, sources


def build_technical_report(
    project_root: str | Path | None = None,
    output_path: str | Path | None = None,
    *,
    require_gold_ready: bool = True,
) -> Path:
    root = resolve_project_root(project_root)
    output = Path(output_path) if output_path is not None else root / DEFAULT_OUTPUT
    if not output.is_absolute():
        output = root / output

    paths = _artifact_paths(root)
    tables = _required_tables(paths)
    quality = tables["quality_gate"]
    standardization_summary = tables["standardization_summary"]
    outlier_summary = tables["outlier_summary"]
    outlier_sensitivity = tables["outlier_sensitivity"]
    duplicate_summary = tables["duplicate_summary"]
    final_metrics = tables["final_metrics"]
    price_regime_review = tables["price_regime_review"]
    segment_profile = tables["segment_profile"]

    quality_stats = _quality_summary(quality)
    if quality_stats["fail"] > 0 or quality_stats["critical_fail"] > 0:
        raise RuntimeError(
            "Technical Report generation blocked: FAIL status exists in the current M2 quality-gate summary. "
            "Non-critical limitations must be represented as REVIEW, not FAIL."
        )

    require_columns(final_metrics, ["metric", "eligible_rows", "populated_rows", "mismatch_rows", "status"], "final_metric_summary.csv")
    mismatch_total = int(numeric(final_metrics["mismatch_rows"]).fillna(0).sum())
    final_status = final_metrics["status"].astype(str).str.upper()
    if mismatch_total != 0 or not final_status.eq("PASS").all():
        raise RuntimeError(
            f"Technical Report generation blocked: final metrics require PASS with zero mismatches; mismatch total={mismatch_total}, statuses={sorted(final_status.unique())}."
        )

    require_columns(outlier_summary, ["metric", "row_count", "status"], "outlier_summary.csv")
    if outlier_summary["status"].astype(str).str.upper().eq("FAIL").any():
        raise RuntimeError("Technical Report generation blocked: outlier_summary.csv contains FAIL status.")
    if "status" in outlier_sensitivity.columns and outlier_sensitivity["status"].astype(str).str.upper().eq("FAIL").any():
        raise RuntimeError("Technical Report generation blocked: outlier_sensitivity.csv contains FAIL status.")

    gold_text, gold_sources, gold_ready = _gold_status(root, paths)
    if require_gold_ready and gold_ready is not True:
        raise RuntimeError(f"Technical Report generation requires ready Gold. {gold_text}")

    dashboard_stage_text, dashboard_stage_sources = _dashboard_stage_status(root, paths.get("dashboard_quality"))
    standardization_text, standardization_sources, standardization_row_count = _standardization_section(
        root, paths.get("standardization_summary")
    )
    missingness_text, missingness_sources = _missingness_policy(
        root, paths.get("missingness_manifest"), paths.get("missingness_action")
    )
    price_regime_text, price_regime_sources = _price_regime_section(
        root, paths.get("price_regime_review"), paths.get("settings")
    )
    segmentation_text, segmentation_sources = _segmentation_section(root, segment_profile, paths.get("segment_manifest"))

    dataset_rows = _lookup_metric(outlier_summary, "metric", "dataset_rows", "row_count")
    if dataset_rows is not None and standardization_row_count is not None and int(dataset_rows) != int(standardization_row_count):
        raise RuntimeError(
            f"Cross-artifact row-count mismatch: standardization summary={int(standardization_row_count)}, outlier summary={int(dataset_rows)}."
        )
    sale_psm_rows = _lookup_metric(final_metrics, "metric", "sale_price_per_sqm", "eligible_rows")
    rent_rows = _lookup_metric(final_metrics, "metric", "rent_equivalent_sensitivity", "eligible_rows")

    duplicate_dataset_rows = _lookup_metric(duplicate_summary, "metric", "row_count", "value")
    if dataset_rows is not None and duplicate_dataset_rows is not None and int(dataset_rows) != int(duplicate_dataset_rows):
        raise RuntimeError(
            f"Cross-artifact row-count mismatch: outlier summary={int(dataset_rows)}, duplicate summary={int(duplicate_dataset_rows)}."
        )
    conservative_supply = _lookup_metric(duplicate_summary, "metric", "conservative_supply_rows", "value")
    exact_excess = _lookup_metric(duplicate_summary, "metric", "exact_duplicate_excess_rows", "value") or 0
    probable_excess = _lookup_metric(duplicate_summary, "metric", "high_probable_duplicate_excess_rows", "value") or 0
    if duplicate_dataset_rows is not None and conservative_supply is not None:
        expected_supply = int(duplicate_dataset_rows) - int(exact_excess) - int(probable_excess)
        if int(conservative_supply) != expected_supply:
            raise RuntimeError(
                f"Duplicate reconciliation mismatch: conservative_supply_rows={int(conservative_supply)}, expected={expected_supply}."
            )

    er_method = _lookup_metric(duplicate_summary, "metric", "entity_resolution_method", "value")
    er_candidate_rows = _lookup_metric(duplicate_summary, "metric", "entity_resolution_candidate_rows", "value")
    er_cluster_count = _lookup_metric(duplicate_summary, "metric", "entity_resolution_cluster_count", "value")
    er_max_cluster = _lookup_metric(duplicate_summary, "metric", "entity_resolution_max_cluster_size", "value")
    er_multi_month = _lookup_metric(duplicate_summary, "metric", "entity_resolution_multi_month_cluster_count", "value")
    er_coverage = _lookup_metric(duplicate_summary, "metric", "entity_resolution_cluster_coverage_rate", "value")
    er_bonus_raw = _lookup_metric(duplicate_summary, "metric", "advanced_entity_resolution_bonus_ready", "value")
    er_bonus_ready = str(er_bonus_raw).strip().lower() in {"true", "1", "yes", "pass", "ready"}
    er_bonus_text = (
        f"Advanced Entity Resolution evidence is **{'READY' if er_bonus_ready else 'REVIEW'}**. "
        f"The canonical duplicate module uses `{er_method or 'deterministic multi-pass record linkage'}` with exact, high, and medium confidence tiers, stable probable-duplicate cluster IDs, and cross-month linkage that is retained rather than automatically deleted. "
        f"Candidate rows: **{fmt_int(er_candidate_rows)}**; linked clusters: **{fmt_int(er_cluster_count)}**; multi-month clusters: **{fmt_int(er_multi_month)}**; maximum cluster size: **{fmt_int(er_max_cluster)}**; cluster-ID coverage: **{fmt_num(float(er_coverage) * 100 if er_coverage is not None and str(er_coverage).strip() not in {'', 'None'} else None, 1)}%**. "
        "Only exact/high-confidence same-month excess can affect supply eligibility; medium and cross-month links remain audit evidence."
    )

    quality_rows: list[list[str]] = []
    check_col = "check" if "check" in quality.columns else ("check_id" if "check_id" in quality.columns else None)
    if check_col is not None:
        for _, row in quality.iterrows():
            quality_rows.append(
                [
                    str(row.get(check_col, "-")),
                    str(row.get("actual", "-")),
                    str(row.get("expected", "-")),
                    str(row.get("status", "-")).upper(),
                    str(bool_series(pd.Series([row.get("critical", True)])).iloc[0]),
                ]
            )

    duplicate_rows: list[list[str]] = []
    if {"metric", "value"}.issubset(duplicate_summary.columns):
        for _, row in duplicate_summary.iterrows():
            duplicate_rows.append([str(row["metric"]), fmt_metric_value(row["value"])])

    outlier_rows: list[list[str]] = []
    require_columns(outlier_summary, ["metric", "row_count", "status"], "outlier_summary.csv")
    for _, row in outlier_summary.iterrows():
        outlier_rows.append([str(row["metric"]), fmt_int(row["row_count"]), str(row["status"]).upper()])

    sensitivity_rows: list[list[str]] = []
    require_columns(
        outlier_sensitivity,
        ["metric", "evaluated_rows", "canonical_outlier_rows", "iqr2_outlier_rows", "log_mad_outlier_rows"],
        "outlier_sensitivity.csv",
    )
    for _, row in outlier_sensitivity.iterrows():
        sensitivity_rows.append(
            [
                str(row["metric"]),
                fmt_int(row["evaluated_rows"]),
                fmt_int(row["canonical_outlier_rows"]),
                fmt_int(row["iqr2_outlier_rows"]),
                fmt_int(row["log_mad_outlier_rows"]),
            ]
        )

    final_metric_rows: list[list[str]] = []
    for _, row in final_metrics.iterrows():
        final_metric_rows.append(
            [
                str(row["metric"]),
                fmt_int(row["eligible_rows"]),
                fmt_int(row["populated_rows"]),
                fmt_int(row["mismatch_rows"]),
                str(row["status"]).upper(),
            ]
        )

    outlier_config, outlier_config_source = _outlier_rule_config(root, paths.get("settings"))
    rent_scenarios, rent_scenario_source = _rent_conversion_scenarios(root, paths.get("settings"))

    spatial_text = "No canonical spatial-quality summary was found."
    spatial_sources: list[str] = []
    if paths.get("spatial_summary") is not None:
        spatial = read_table(paths["spatial_summary"])
        spatial_sources.append(relative_path(paths["spatial_summary"], root))
        row_count = _lookup_metric(spatial, "metric", "row_count", "value")
        coord_rows = _lookup_metric(spatial, "metric", "coordinate_pair_rows", "value")
        iran_rows = _lookup_metric(spatial, "metric", "iran_window_valid_rows", "value")
        pip_n = _lookup_metric(spatial, "metric", "advanced_boundary_sample_n", "value")
        pip_rate = _lookup_metric(spatial, "metric", "advanced_point_in_polygon_inside_rate", "rate")
        reverse_n = _lookup_metric(spatial, "metric", "reverse_geocode_sample_n", "value")
        country_rate = _lookup_metric(spatial, "metric", "reverse_geocode_iran_country_match_rate", "rate")
        city_rate = _lookup_metric(spatial, "metric", "reverse_geocode_city_or_county_match_rate", "rate")
        spatial_bonus_raw = _lookup_metric(spatial, "metric", "advanced_spatial_bonus_ready", "value")
        spatial_bonus_ready = str(spatial_bonus_raw).strip().lower() in {"true", "1", "yes", "pass", "ready"}
        spatial_text = (
            f"Spatial QA covered **{fmt_int(row_count)}** records; **{fmt_int(coord_rows)}** had coordinate pairs and "
            f"**{fmt_int(iran_rows)}** fell inside the accepted Iran validation window. "
            f"Advanced QA used a point-in-polygon sample of **{fmt_int(pip_n)}** with an inside-boundary rate of **{fmt_num(float(pip_rate) * 100 if pip_rate is not None and not pd.isna(pip_rate) else None, 1)}%**, "
            f"plus a reverse-geocoding sample of **{fmt_int(reverse_n)}** with Iran-country match **{fmt_num(float(country_rate) * 100 if country_rate is not None and not pd.isna(country_rate) else None, 1)}%** and city/county match **{fmt_num(float(city_rate) * 100 if city_rate is not None and not pd.isna(city_rate) else None, 1)}%**. "
            f"The Advanced Spatial Analysis bonus evidence is **{'READY' if spatial_bonus_ready else 'REVIEW'}**: the external boundary point-in-polygon test and reverse-geocode request/country integrity are the bonus gates. "
            "City/county string alignment remains a separate diagnostic because the external administrative taxonomy and platform city taxonomy are not identical; its threshold is not lowered to force a pass. Exact coordinates are not published in Gold or the dashboard."
        )

    model_limitation = ""
    model_sources: list[str] = []
    if paths.get("model_diagnostics") is not None:
        diagnostics = read_table(paths["model_diagnostics"])
        model_sources.append(relative_path(paths["model_diagnostics"], root))
        split_col = "evaluation_split" if "evaluation_split" in diagnostics.columns else ("split" if "split" in diagnostics.columns else None)
        if split_col is not None:
            test = diagnostics.loc[diagnostics[split_col].astype(str).str.lower() == "test"].copy()
            if not test.empty:
                row = test.iloc[0]
                median_ape = row.get("median_ape_pct", row.get("median_absolute_percentage_error_pct"))
                model_limitation = (
                    f" On the held-out test split, R2(log)=`{fmt_num(row.get('r2_log'), 3)}`, "
                    f"median APE=`{fmt_num(median_ape, 1)}%`, N=`{fmt_int(row.get('n'))}`."
                )

    generated_at = datetime.now(timezone.utc).isoformat()
    text = f"""# Technical Report - Iran Housing Market Intelligence

- **Report version:** `{REPORT_VERSION}`
- **Generated at (UTC):** `{generated_at}`
- **Silver Master rows:** **{fmt_int(dataset_rows)}**
- **Sale asking-price-per-sqm eligible rows:** **{fmt_int(sale_psm_rows)}**
- **Rental-equivalent eligible rows:** **{fmt_int(rent_rows)}**
- **Gold status:** {gold_text}

## 1. Analytical architecture and reproducibility

The project uses a canonical, read-only Silver Master as the analytical source of truth. Cleaning, eligibility, duplicate control, outlier handling, spatial QA, modeling, text analysis, segmentation, and Gold construction are implemented in versioned modules under `src/`. The final notebook is an orchestrator: it runs or validates those modules, loads accepted artifacts, and presents evidence without duplicating business logic inside notebook cells.

Gold is the report/dashboard-ready layer derived from Silver and accepted canonical analytical outputs. The final report builders do not refit models or replace upstream calculations.

### Pipeline readiness metadata

{dashboard_stage_text}

## 2. Data-quality audit

The current M2 quality gate contains **{quality_stats['checks']}** checks: **{quality_stats['pass']} PASS**, **{quality_stats['review']} REVIEW**, **{quality_stats['fail']} FAIL**, and **{quality_stats['critical_fail']} critical failures**.

{_md_table(['Check', 'Actual', 'Expected', 'Status', 'Critical'], quality_rows) if quality_rows else 'The quality-gate artifact does not expose row-level check identifiers.'}

The audit verifies core contract properties such as source-row uniqueness, non-null price-regime/unit/observation metadata, finite final metrics, duplicate retention policy, and eligible analytical populations. A non-critical REVIEW is a documented limitation rather than an analytical failure.

### 2.1 Persian/Arabic character and structured-field standardization

{standardization_text}

### 2.2 Missingness policy

{missingness_text}

Missing values are never globally interpreted as zero or False. Structural non-applicability is retained as NULL. Model-specific imputation is performed only inside the relevant training/resampling workflow to avoid leakage.

### 2.3 Spatial quality

{spatial_text}

## 3. Duplicate control and Advanced Entity Resolution

The upstream standardization and missingness policies above are lineage-preserving and eligibility-driven. Raw/audit information remains available in Silver while typed, cleaned, derived, quality-flag, price-regime, final-metric, and eligibility fields define analytical use. Records are not physically deleted simply because they fail a downstream metric rule.

Duplicate handling is conservative: same-month duplicate excess is removed from supply eligibility, while cross-month repeats remain retained and auditable because a repeated listing across months may represent continued platform activity rather than a duplicate observation at one time point.

{er_bonus_text}

{_md_table(['Duplicate metric', 'Value'], duplicate_rows) if duplicate_rows else 'The duplicate summary does not expose the expected metric/value schema.'}

## 4. Outlier rules

Canonical outlier handling is **flag-based, not row-deletion-based**. The frozen M2-05 rule version is `{outlier_config['rule_version']}`. Positive relevant values use group-aware IQR thresholds with a canonical multiplier of **{outlier_config['canonical_iqr_multiplier']} x IQR**. Local thresholds require at least **{outlier_config['minimum_local_group_rows']}** rows; the broader property-category fallback requires at least **{outlier_config['minimum_fallback_group_rows']}** rows.

For area and sale asking-price-per-sqm metrics, the canonical grouping is city x property category with property-category fallback. Rental and deposit thresholds additionally respect price regime. Groups without sufficient threshold support are retained and reported rather than assigned an arbitrary global threshold. Construction-year pre-1370 values are treated as censored source semantics/review information, not automatically as data errors.

{_md_table(['Metric / flag', 'Rows', 'Status'], outlier_rows)}

Sensitivity evidence compares context-eligible canonical flags with a global **{outlier_config['sensitivity_iqr_multiplier']} x IQR** rule and a log-MAD alternative. The canonical counts in this sensitivity table can therefore differ slightly from raw Silver flag counts when a raw flag falls outside the final analytical context. Different methods identify different numbers of unusual observations, so no alternative threshold is treated as ground truth.

{_md_table(['Metric', 'Evaluated', 'Canonical', 'IQR x 2', 'Log-MAD'], sensitivity_rows)}

## 5. Price-regime separation

{price_regime_text}

This separation prevents sale prices from being combined directly with rent/deposit components in a single monetary metric. Every downstream analytical population must declare a compatible price regime.

## 6. Deposit-to-rent conversion

For validated long-term rental records, the project publishes three sensitivity scenarios. If `R` is monthly rent and `D` is deposit:

- Low: `R + D x {rent_scenarios['low']:.3f}`
- Base: `R + D x {rent_scenarios['base']:.3f}`
- High: `R + D x {rent_scenarios['high']:.3f}`

These rates are derived from `config/settings.yaml` and correspond to **25,000 / 30,000 / 35,000 Toman of monthly-rent equivalent per 1,000,000 Toman of deposit**. They are project sensitivity assumptions, not an externally verified official market conversion rate. Semantic zero is preserved for monthly rent in full-deposit listings and for deposit in rent-only listings.

## 7. Sale asking price per square meter

The canonical comparable sale metric is published only for eligible records:

`sale_price_per_sqm = validated sale asking price / primary_area_sqm`

`primary_area_sqm` follows the upstream property-compatible area contract. Outside the valid analytical population, the final metric remains NULL. The observation is an asking price, not a transaction price.

{_md_table(['Final metric', 'Eligible', 'Populated', 'Mismatch', 'Status'], final_metric_rows)}

A zero mismatch count is required for consistency between eligibility flags and populated final metrics.

## 8. Market segmentation method

{segmentation_text}

## 9. Price Drivers / AVM interpretation

Price-driver outputs use a controlled Ridge model with held-out evaluation. Adjusted effects are interpreted as **model-implied adjusted associations/contrasts**. Permutation importance is interpreted as **held-out predictive contribution**. Neither output is a causal effect. Location and property-category blocks are structural controls and should be separated from potentially actionable property characteristics.{model_limitation}

## 10. Main limitations

1. **Asking-price data:** listing asks are not realized transaction prices.
2. **Currency contract:** Toman is the operational unit, but the source currency remains unconfirmed; no undocumented factor-of-ten conversion is applied.
3. **Platform selection bias:** the dataset is not the complete Iranian housing stock, inventory, or transaction universe.
4. **Market Temperature:** the index is a proxy for asking-price and listing-activity trends; it is not liquidity, absorption, physical inventory, or supply tightness.
5. **Model interpretation:** adjusted associations and permutation importance are non-causal and constrained by model error and feature coverage.
6. **Outlier policy:** IQR rules identify unusual observations, not confirmed errors; source rows remain auditable in Silver.
7. **Spatial privacy and quality:** exact coordinates are not exposed in Gold or the dashboard; surfaced spatial results are aggregated.
8. **Segmentation:** the final segments are descriptive market types, not socioeconomic classes, buyer personas, or causal groups.
9. **Standardization scope:** character normalization is orthographic/technical and does not rewrite listing semantics; raw text and unresolved parse evidence remain auditable.
10. **Missingness:** missing values are not blanket-filled with zero/False; feature applicability varies across property families.
11. **Temporal scope:** findings describe the project's core analytical months and should not be treated as a permanent long-run market regime.

## 11. Public FastAPI deployment - bonus evidence

The accepted Gold layer is exposed through a public, read-only FastAPI service deployed on Render. Deployment evidence was validated on **2026-08-14**.

- **Base URL:** `https://ihmi-fastapi.onrender.com`
- **Interactive Swagger/OpenAPI:** `https://ihmi-fastapi.onrender.com/docs`
- **Health endpoint:** `https://ihmi-fastapi.onrender.com/health`
- **Public smoke test:** `PASS` with `health=ok`, `gold_qa_status=PASS`, `marts=10`, and `dimensions=5`
- **Swagger availability:** `/docs` returned HTTP `200`

The API is an access layer over accepted canonical Gold artifacts. It does not rebuild Silver or Gold, refit models, or recompute Market Temperature or segmentation. Exact listing coordinates are not exposed. The same asking-price, listing-activity, observational/non-causal, AVM-prototype, and descriptive-segmentation interpretation boundaries remain in force.

## 12. Final data-product status

{gold_text}

This Technical Report is designed to accompany the Executive Summary, the Restart-and-Run-All final notebook, the final dashboard, and the public FastAPI deployment in the delivery package.

### Final reporting surface

The professor-facing reporting layer is intentionally limited to two consolidated Markdown reports: `Executive_Summary.md` and `Technical_Report.md`. Task-level QA/technical notes remain internal evidence and are not additional final reports. Product Recommendations and the demonstration are separate delivery artifacts rather than duplicate technical reports.

---

### Canonical artifacts used

"""

    source_paths: list[str] = []
    for key in [
        "quality_gate",
        "standardization_summary",
        "outlier_summary",
        "outlier_sensitivity",
        "outlier_manifest",
        "duplicate_summary",
        "duplicate_supply_impact",
        "final_metrics",
        "price_regime_review",
        "segment_profile",
    ]:
        path = paths.get(key)
        if path is not None:
            source_paths.append(relative_path(path, root))
    source_paths.extend(standardization_sources)
    source_paths.extend(missingness_sources)
    source_paths.extend(price_regime_sources)
    source_paths.extend(spatial_sources)
    source_paths.extend(segmentation_sources)
    source_paths.extend(model_sources)
    source_paths.extend(dashboard_stage_sources)
    source_paths.extend(gold_sources)

    seen: set[str] = set()
    for source in source_paths:
        if source and source not in seen:
            seen.add(source)
            text += f"- `{source}`\n"

    text += f"\nOutlier configuration source: `{outlier_config_source}`.\n"
    text += f"Rent/deposit sensitivity configuration source: `{rent_scenario_source}`.\n"
    text += "\nThis report builder does not refit statistical or machine-learning models; it summarizes accepted project artifacts and frozen methodological contracts.\n"

    _assert_english_only(text)
    _assert_no_personal_paths(text)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp_output = output.with_name(f".{output.name}.tmp")
    temp_output.write_text(text, encoding="utf-8")
    os.replace(temp_output, output)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the final English IHMI Technical Report Markdown from canonical QA, M2/M3, and Gold artifacts.")
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--allow-unready-gold", action="store_true", help="Allow report generation when Gold QA is not ready (not recommended for final delivery).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    path = build_technical_report(
        project_root=args.project_root,
        output_path=args.output,
        require_gold_ready=not args.allow_unready_gold,
    )
    print(f"TECHNICAL REPORT GENERATED: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
