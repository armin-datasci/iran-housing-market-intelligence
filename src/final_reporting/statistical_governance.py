from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import polars as pl

from src.common.io_utils import atomic_write_csv, atomic_write_json
from src.common.paths import OUTPUTS_DIR, relative_to_project
from src.common.validation import Check, checks_frame, make_check, summarize_checks


VERSION = "final-statistical-governance-v1.0"
PROJECT_ROOT = Path(__file__).resolve().parents[2]

TABLE_DIR = OUTPUTS_DIR / "tables" / "final_reporting"
QA_DIR = OUTPUTS_DIR / "qa" / "final_reporting"

METHOD_MAP_PATH = TABLE_DIR / "statistical_method_map.csv"
ROBUSTNESS_PATH = TABLE_DIR / "robustness_sensitivity_matrix.csv"
READINESS_PATH = TABLE_DIR / "statistical_readiness_matrix.csv"
DESIGN_SUMMARY_PATH = TABLE_DIR / "statistical_design_summary.csv"
CHECKS_PATH = QA_DIR / "statistical_governance_checks.csv"
MANIFEST_PATH = QA_DIR / "statistical_governance_manifest.json"

M2_TABLES = OUTPUTS_DIR / "tables" / "milestone_2"
M3_TABLES = OUTPUTS_DIR / "tables" / "milestone_3"
M3_QA = OUTPUTS_DIR / "qa" / "milestone_3"

ARTIFACTS: dict[str, tuple[Path, ...]] = {
    "quality_gate": (
        M2_TABLES / "quality_gate" / "quality_gate_summary.csv",
    ),
    "duplicate_summary": (
        M2_TABLES / "duplicates" / "duplicate_summary.csv",
    ),
    "duplicate_supply_impact": (
        M2_TABLES / "duplicates" / "duplicate_supply_impact.csv",
    ),
    "outlier_summary": (
        M2_TABLES / "outliers" / "outlier_summary.csv",
    ),
    "outlier_sensitivity": (
        M2_TABLES / "outliers" / "outlier_sensitivity.csv",
    ),
    "final_metrics": (
        M2_TABLES / "final_metrics" / "final_metric_summary.csv",
    ),
    "monthly_price": (
        M3_TABLES / "monthly_market" / "monthly_price_summary.csv",
    ),
    "market_temperature_sensitivity": (
        M3_QA / "market_temperature" / "market_temperature_sensitivity.csv",
        M3_TABLES / "market_temperature" / "market_temperature_sensitivity.csv",
    ),
    "market_temperature_manifest": (
        M3_QA / "market_temperature" / "market_temperature_manifest.json",
        M3_TABLES / "market_temperature" / "market_temperature_manifest.json",
    ),
    "price_driver_diagnostics": (
        M3_TABLES / "price_drivers" / "price_driver_model_diagnostics.csv",
    ),
    "price_driver_importance": (
        M3_TABLES / "price_drivers" / "price_driver_permutation_importance.csv",
    ),
    "price_driver_manifest": (
        M3_QA / "price_drivers" / "price_driver_manifest.json",
        M3_TABLES / "price_drivers" / "price_driver_manifest.json",
    ),
    "seller_summary": (
        M3_TABLES / "seller_type_comparison" / "seller_type_comparison_summary.csv",
    ),
    "seller_manifest": (
        M3_QA / "seller_type_comparison" / "seller_type_manifest.json",
        M3_QA / "seller_type_comparison" / "seller_type_comparison_manifest.json",
    ),
    "text_summary": (
        M3_TABLES / "text_analysis" / "text_signal_summary.csv",
    ),
    "text_precision": (
        M3_QA / "text_analysis" / "keyword_precision_summary.csv",
        M3_TABLES / "text_analysis" / "keyword_precision_summary.csv",
    ),
    "text_manifest": (
        M3_QA / "text_analysis" / "text_signal_manifest.json",
        M3_TABLES / "text_analysis" / "text_signal_manifest.json",
    ),
    "segment_profile": (
        M3_TABLES / "market_segmentation" / "segment_profile.csv",
    ),
    "segment_manifest": (
        M3_QA / "market_segmentation" / "segmentation_manifest.json",
        M3_TABLES / "market_segmentation" / "segmentation_manifest.json",
    ),
    "cluster_diagnostics": (
        M3_QA / "market_segmentation" / "cluster_selection_diagnostics.csv",
        M3_TABLES / "market_segmentation" / "cluster_selection_diagnostics.csv",
    ),
    "spatial_summary": (
        M3_TABLES / "spatial_quality" / "spatial_quality_summary.csv",
    ),
    "gold_price_regime": (
        PROJECT_ROOT / "data" / "gold" / "dimensions" / "dim_price_regime.parquet",
    ),
    "gold_qa_manifest": (
        PROJECT_ROOT / "data" / "gold" / "qa" / "gold_qa_manifest.json",
    ),
    "technical_report": (
        PROJECT_ROOT / "reports" / "final" / "Technical_Report.md",
    ),
    "data_dictionary": (
        PROJECT_ROOT / "docs" / "data_dictionary.csv",
        OUTPUTS_DIR / "tables" / "milestone_1" / "data_dictionary.csv",
        OUTPUTS_DIR / "tables" / "data_dictionary.csv",
    ),
    "settings": (
        PROJECT_ROOT / "config" / "settings.yaml",
        PROJECT_ROOT / "config" / "settings.yml",
    ),
}


def _first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _artifact_path(name: str) -> Path | None:
    return _first_existing(ARTIFACTS[name])


def _artifact_ref(name: str) -> str:
    path = _artifact_path(name)
    if path is not None:
        return relative_to_project(path)
    return "MISSING: " + " | ".join(relative_to_project(path) for path in ARTIFACTS[name])


def _artifact_exists(name: str) -> bool:
    return _artifact_path(name) is not None


def _read_csv(name: str) -> pd.DataFrame:
    path = _artifact_path(name)
    if path is None:
        return pd.DataFrame()
    return pd.read_csv(path)


def _read_json(name: str) -> dict[str, Any]:
    path = _artifact_path(name)
    if path is None:
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        return {}


def _nested_get(payload: dict[str, Any], *paths: tuple[str, ...]) -> Any:
    for path in paths:
        current: Any = payload
        valid = True
        for key in path:
            if not isinstance(current, dict) or key not in current:
                valid = False
                break
            current = current[key]
        if valid:
            return current
    return None


def _first_value(frame: pd.DataFrame, column: str) -> Any:
    if frame.empty or column not in frame.columns:
        return None
    values = frame[column].dropna()
    return values.iloc[0] if not values.empty else None


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _fmt_pct(value: Any, digits: int = 1) -> str:
    number = _safe_float(value)
    return "n/a" if number is None else f"{number:+.{digits}f}%"


def _combine_evidence(*artifact_names: str) -> str:
    return "; ".join(_artifact_ref(name) for name in artifact_names)


def _status_from_evidence(*artifact_names: str, review: bool = False) -> str:
    if not all(_artifact_exists(name) for name in artifact_names):
        return "FAIL"
    return "REVIEW" if review else "PASS"


def _currency_status() -> tuple[str, str]:
    quality = _read_csv("quality_gate")
    if quality.empty:
        return "FAIL", "Quality-gate evidence is unavailable."

    row = pd.DataFrame()
    for key_col in ("check_id", "metric", "check"):
        if key_col in quality.columns:
            mask = quality[key_col].astype(str).str.contains(
                "currency_source_confirmation", case=False, na=False
            )
            row = quality.loc[mask]
            if not row.empty:
                break

    if row.empty:
        return (
            "REVIEW",
            "Operational Toman contract is retained, but the currency-source review row was not found.",
        )

    actual = None
    for value_col in ("actual", "value", "metric_value"):
        if value_col in row.columns:
            actual = row.iloc[0][value_col]
            break

    actual_text = str(actual or "").strip()
    if actual_text == "toman_assumed_unconfirmed":
        return (
            "REVIEW",
            "Operational unit is Toman; source currency remains unconfirmed and no factor-of-ten conversion is applied.",
        )
    return "REVIEW", f"Currency-source status requires interpretation: {actual_text or 'unknown'}."


def _segmentation_context() -> dict[str, Any]:
    manifest = _read_json("segment_manifest")
    primary_method = _nested_get(
        manifest,
        ("primary_method",),
        ("method", "primary_method"),
        ("definition", "primary_method"),
    )
    methodology_status = _nested_get(
        manifest,
        ("methodology_status",),
        ("status", "methodology_status"),
    )
    return {
        "primary_method": str(primary_method or "unknown"),
        "methodology_status": str(methodology_status or "unknown"),
        "cluster_diagnostics_available": _artifact_exists("cluster_diagnostics"),
    }


def _seller_context() -> dict[str, Any]:
    frame = _read_csv("seller_summary")
    return {
        "raw": _first_value(frame, "raw_median_difference_pct"),
        "adjusted": _first_value(frame, "adjusted_crossfit_difference_pct"),
        "adjusted_low": _first_value(frame, "adjusted_ci_low_pct"),
        "adjusted_high": _first_value(frame, "adjusted_ci_high_pct"),
        "stratified": _first_value(frame, "stratified_difference_pct"),
        "agency_n": _first_value(frame, "agency_n"),
        "personal_n": _first_value(frame, "personal_n"),
    }


def _text_context() -> dict[str, Any]:
    frame = _read_csv("text_summary")
    if frame.empty:
        return {"tested": 0, "fdr_significant": 0, "validated": 0}

    analyzed = frame.get(
        "analysis_status", pd.Series(index=frame.index, dtype=object)
    ).astype(str).eq("PASS")
    fdr = frame.get(
        "adjusted_fdr_bh_status", pd.Series(index=frame.index, dtype=object)
    ).astype(str).eq("SIGNIFICANT_FDR_0_05")
    validated = frame.get(
        "precision_validation_status", pd.Series(index=frame.index, dtype=object)
    ).astype(str).eq("PASS")

    return {
        "tested": int(analyzed.sum()),
        "fdr_significant": int((analyzed & fdr).sum()),
        "validated": int(validated.sum()),
    }


def build_statistical_method_map() -> pd.DataFrame:
    seg = _segmentation_context()
    rule_based_release = "rule" in seg["primary_method"].lower()
    cluster_evidence = seg["cluster_diagnostics_available"]

    rows = [
        {
            "source_section": "16 - Clustering and Segmentation",
            "concept": "Standardization / scaling",
            "project_role": "Make candidate clustering features comparable without raw scale dominating.",
            "implementation_status": "IMPLEMENTED_IN_SEGMENTATION_EVALUATION",
            "implementation": "SEG-27 uses documented feature transformation/scaling inside its segmentation workflow; the final release remains method-aware.",
            "canonical_evidence": _combine_evidence("segment_manifest", "cluster_diagnostics"),
            "interpretation_boundary": "Scaling is preprocessing, not evidence that the released market types are natural latent classes.",
        },
        {
            "source_section": "16 - Clustering and Segmentation",
            "concept": "MiniBatchKMeans / K-means",
            "project_role": "Evaluate whether a stable unsupervised apartment-sale segmentation exists.",
            "implementation_status": (
                "EVALUATED_NOT_RELEASED_AS_PRIMARY"
                if rule_based_release and cluster_evidence
                else "EVIDENCE_REVIEW"
            ),
            "implementation": f"Candidate clustering was evaluated; accepted manifest primary_method={seg['primary_method']}.",
            "canonical_evidence": _combine_evidence("segment_manifest", "cluster_diagnostics"),
            "interpretation_boundary": "Do not label the released descriptive typology as K-means clusters when the stability gate did not support that claim.",
        },
        {
            "source_section": "16 - Clustering and Segmentation",
            "concept": "Silhouette score",
            "project_role": "Candidate-k separation diagnostic.",
            "implementation_status": "IMPLEMENTED_DIAGNOSTIC" if cluster_evidence else "EVIDENCE_REVIEW",
            "implementation": "Used with additional quality and stability criteria; never used alone to force a cluster count.",
            "canonical_evidence": _combine_evidence("cluster_diagnostics", "segment_manifest"),
            "interpretation_boundary": "A higher score alone is insufficient for release if cluster-share or stability gates fail.",
        },
        {
            "source_section": "16 - Clustering and Segmentation",
            "concept": "Davies-Bouldin index",
            "project_role": "Candidate-k compactness/separation diagnostic.",
            "implementation_status": "IMPLEMENTED_DIAGNOSTIC" if cluster_evidence else "EVIDENCE_REVIEW",
            "implementation": "Evaluated jointly with Silhouette, Calinski-Harabasz, minimum cluster share and resampling stability.",
            "canonical_evidence": _combine_evidence("cluster_diagnostics", "segment_manifest"),
            "interpretation_boundary": "Lower DBI supports comparison but does not guarantee business-segment validity.",
        },
        {
            "source_section": "16 - Clustering and Segmentation",
            "concept": "Calinski-Harabasz score",
            "project_role": "Project-specific extension to the notebook's clustering diagnostics.",
            "implementation_status": "IMPLEMENTED_DIAGNOSTIC" if cluster_evidence else "EVIDENCE_REVIEW",
            "implementation": "Included in candidate-k quality diagnostics.",
            "canonical_evidence": _combine_evidence("cluster_diagnostics", "segment_manifest"),
            "interpretation_boundary": "Used as one diagnostic among several, not as a standalone release gate.",
        },
        {
            "source_section": "16 - Clustering and Segmentation",
            "concept": "Resampling ARI stability",
            "project_role": "Test assignment stability across resampling/seed perturbations.",
            "implementation_status": "IMPLEMENTED_STABILITY_GATE" if cluster_evidence else "EVIDENCE_REVIEW",
            "implementation": "Pairwise Adjusted Rand Index is part of the predeclared stability criterion.",
            "canonical_evidence": _combine_evidence("cluster_diagnostics", "segment_manifest"),
            "interpretation_boundary": "Thresholds must not be lowered post hoc to force a clustering release.",
        },
        {
            "source_section": "16 - Clustering and Segmentation",
            "concept": "PCA",
            "project_role": "Potential dimensionality reduction.",
            "implementation_status": "NOT_USED",
            "implementation": "Not required in the accepted segmentation path; interpretability of original structural features is preserved.",
            "canonical_evidence": _artifact_ref("segment_manifest"),
            "interpretation_boundary": "Do not claim PCA was used unless a future accepted SEG-27 artifact explicitly records it.",
        },
        {
            "source_section": "16 - Clustering and Segmentation",
            "concept": "Hierarchical clustering (HAC)",
            "project_role": "Alternative clustering family.",
            "implementation_status": "NOT_USED",
            "implementation": "Not part of the accepted SEG-27 release.",
            "canonical_evidence": _artifact_ref("segment_manifest"),
            "interpretation_boundary": "Do not add a parallel clustering method solely for checklist completeness.",
        },
        {
            "source_section": "16 - Clustering and Segmentation",
            "concept": "Final released market segmentation",
            "project_role": "Professor-facing interpretable market types.",
            "implementation_status": (
                "RELEASED_DESCRIPTIVE_TYPOLOGY" if rule_based_release else "REVIEW_METHOD"
            ),
            "implementation": f"Accepted release uses primary_method={seg['primary_method']}; segments are profiled and named from observed listing characteristics.",
            "canonical_evidence": _combine_evidence("segment_profile", "segment_manifest"),
            "interpretation_boundary": "Descriptive market types are not causal classes, socioeconomic classes or verified buyer personas.",
        },
        {
            "source_section": "20 - Housing-specific concepts",
            "concept": "Price per Square Meter (PSM)",
            "project_role": "Canonical comparable sale asking-price metric.",
            "implementation_status": "IMPLEMENTED",
            "implementation": "Eligible sale asking price is divided by a valid property-compatible area denominator.",
            "canonical_evidence": _artifact_ref("final_metrics"),
            "interpretation_boundary": "PSM is based on listing asking price, not realized transaction price.",
        },
        {
            "source_section": "20 - Housing-specific concepts",
            "concept": "Hedonic pricing model (HPM)",
            "project_role": "Conceptual framing for controlled asking-price modeling.",
            "implementation_status": "RELATED_IMPLEMENTATION",
            "implementation": "The project uses a controlled Ridge asking-PSM model with held-out evaluation rather than claiming a classical inferential hedonic OLS specification.",
            "canonical_evidence": _combine_evidence("price_driver_diagnostics", "price_driver_manifest"),
            "interpretation_boundary": "Adjusted contrasts are model-implied observational associations, not causal hedonic effects.",
        },
        {
            "source_section": "20 - Housing-specific concepts",
            "concept": "Fixed effects (FE)",
            "project_role": "Control systematic location/time differences.",
            "implementation_status": "CONTROL_ANALOGUE_NOT_FORMAL_PANEL_FE",
            "implementation": "City/neighborhood and time are represented as model control blocks; no formal panel fixed-effects estimator is claimed.",
            "canonical_evidence": _combine_evidence("price_driver_manifest", "price_driver_importance"),
            "interpretation_boundary": "Use the term location/time controls unless a formal FE estimator is explicitly fitted.",
        },
        {
            "source_section": "20 - Housing-specific concepts",
            "concept": "Spatial autocorrelation",
            "project_role": "Relevant spatial-statistics concept for nearby housing markets.",
            "implementation_status": "CONCEPT_RELEVANT_NOT_ESTIMATED",
            "implementation": "Accepted spatial work focuses on coordinate QA, external boundary validation and aggregated geographic analysis.",
            "canonical_evidence": _artifact_ref("spatial_summary"),
            "interpretation_boundary": "Do not claim a spatial-autocorrelation statistic was estimated unless it exists in an accepted artifact.",
        },
        {
            "source_section": "20 - Housing-specific concepts",
            "concept": "Moran's I",
            "project_role": "Potential statistic for spatial autocorrelation.",
            "implementation_status": "NOT_USED",
            "implementation": "Not part of the accepted canonical project outputs.",
            "canonical_evidence": _artifact_ref("spatial_summary"),
            "interpretation_boundary": "No Moran's I result should be reported or implied.",
        },
    ]
    return pd.DataFrame(rows)


def build_robustness_matrix() -> pd.DataFrame:
    seller = _seller_context()
    seg = _segmentation_context()

    rows = [
        {
            "analysis": "Outlier identification",
            "primary_specification": "Group-aware 3x IQR flagging; rows are retained and flagged.",
            "alternative_specification": "2x IQR and log-MAD sensitivity rules.",
            "robustness_type": "Threshold / method sensitivity",
            "status": _status_from_evidence("outlier_summary", "outlier_sensitivity"),
            "canonical_evidence": _combine_evidence("outlier_summary", "outlier_sensitivity"),
            "observed_conclusion": "Alternative rules identify different unusual-observation counts; no alternative is treated as ground truth.",
            "claim_boundary": "Outlier flags indicate unusual observations, not confirmed data errors.",
        },
        {
            "analysis": "Rent/deposit equivalent",
            "primary_specification": "Base monthly-equivalent deposit conversion rate = 0.030.",
            "alternative_specification": "Low=0.025 and High=0.035 scenarios.",
            "robustness_type": "Assumption sensitivity",
            "status": _status_from_evidence("final_metrics", "settings"),
            "canonical_evidence": _combine_evidence("final_metrics", "settings"),
            "observed_conclusion": "Rental-equivalent outputs are scenario-based project assumptions.",
            "claim_boundary": "Conversion factors are not externally verified official market rates.",
        },
        {
            "analysis": "Listing supply / duplicate handling",
            "primary_specification": "Conservative deduplicated listing activity excludes exact/high-confidence same-month excess only.",
            "alternative_specification": "Raw listing activity; medium candidates and cross-month repeats remain retained/auditable.",
            "robustness_type": "Raw-vs-deduplicated robustness",
            "status": _status_from_evidence("duplicate_summary", "duplicate_supply_impact"),
            "canonical_evidence": _combine_evidence("duplicate_summary", "duplicate_supply_impact"),
            "observed_conclusion": "Conservative duplicate removal has a very small aggregate effect on the monthly listing-activity trend.",
            "claim_boundary": "Listing activity is a platform-flow proxy, not physical housing inventory.",
        },
        {
            "analysis": "Listing-Market Temperature Proxy",
            "primary_specification": "Frozen primary price/listing-activity weights, percentile gates and reliability requirements.",
            "alternative_specification": "Predeclared alternative weights/thresholds from the sensitivity artifact.",
            "robustness_type": "Weight / threshold sensitivity",
            "status": _status_from_evidence("market_temperature_manifest", "market_temperature_sensitivity"),
            "canonical_evidence": _combine_evidence("market_temperature_manifest", "market_temperature_sensitivity"),
            "observed_conclusion": "Sensitivity evidence assesses label/ranking stability without redefining the canonical proxy after seeing results.",
            "claim_boundary": "Temperature is not liquidity, absorption, inventory or supply tightness.",
        },
        {
            "analysis": "Seller-type asking-PSM comparison",
            "primary_specification": "Control-only Ridge cross-fit residual contrast.",
            "alternative_specification": "Raw median comparison and coarsened similar-unit stratified robustness check.",
            "robustness_type": "Alternative adjustment strategy",
            "status": _status_from_evidence("seller_summary", "seller_manifest"),
            "canonical_evidence": _combine_evidence("seller_summary", "seller_manifest"),
            "observed_conclusion": (
                f"Raw={_fmt_pct(seller['raw'])}; adjusted cross-fit={_fmt_pct(seller['adjusted'])}; "
                f"similar-unit strata={_fmt_pct(seller['stratified'])}."
            ),
            "claim_boundary": "Seller type is observational; adjusted estimates do not establish a causal agency premium.",
        },
        {
            "analysis": "Market segmentation",
            "primary_specification": f"Accepted primary method: {seg['primary_method']}.",
            "alternative_specification": "MiniBatchKMeans candidate evaluation with quality and resampling-stability gates.",
            "robustness_type": "Method-selection / stability robustness",
            "status": _status_from_evidence("segment_profile", "segment_manifest", "cluster_diagnostics"),
            "canonical_evidence": _combine_evidence("segment_profile", "segment_manifest", "cluster_diagnostics"),
            "observed_conclusion": "A descriptive typology is retained when unsupervised clustering does not satisfy the predeclared release gate.",
            "claim_boundary": "Do not call final market types stable clusters unless the accepted manifest says clustering is primary.",
        },
        {
            "analysis": "Price-driver predictive contribution",
            "primary_specification": "Held-out permutation importance with City + Neighborhood grouped as one location block.",
            "alternative_specification": "Interpret property features separately from structural model-control blocks.",
            "robustness_type": "Grouped-feature permutation design",
            "status": _status_from_evidence("price_driver_importance", "price_driver_manifest"),
            "canonical_evidence": _combine_evidence("price_driver_importance", "price_driver_manifest"),
            "observed_conclusion": "Grouped location permutation avoids impossible city-neighborhood combinations during importance evaluation.",
            "claim_boundary": "Permutation importance is predictive contribution, not causal importance.",
        },
        {
            "analysis": "Validated listing-text signals",
            "primary_specification": "Held-out control-model residual comparison on manually precision-validated keyword families.",
            "alternative_specification": "Benjamini-Hochberg FDR correction across the validated test family.",
            "robustness_type": "Multiple-testing / evidence-control robustness",
            "status": _status_from_evidence("text_summary", "text_precision", "text_manifest"),
            "canonical_evidence": _combine_evidence("text_summary", "text_precision", "text_manifest"),
            "observed_conclusion": "Professor-facing significance labels use q<0.05 after BH-FDR rather than raw p-values alone.",
            "claim_boundary": "Text associations are observational; manual validation estimates precision of detected matches, not recall.",
        },
    ]
    return pd.DataFrame(rows)


def build_readiness_matrix() -> pd.DataFrame:
    currency_status, currency_note = _currency_status()

    rows = [
        ("25 - Shared checklist", "Population and Sample are defined", "PASS",
         "Task-specific analytical populations and sample sizes are declared in canonical outputs/manifests.",
         _combine_evidence("final_metrics", "price_driver_diagnostics", "segment_manifest"),
         "Different tasks use different eligible populations; do not imply one universal analytical N."),
        ("25 - Shared checklist", "Unit of Analysis is explicit", "PASS",
         "Listing-level and aggregate market grains are declared by the relevant task/output contract.",
         _combine_evidence("segment_manifest", "market_temperature_manifest"),
         "Never aggregate mixed grains as though they were one row type."),
        ("25 - Shared checklist", "Sampling / coverage biases are documented",
         "PASS" if _artifact_exists("technical_report") else "REVIEW",
         "Platform selection/coverage limitations belong in final claim boundaries.",
         _artifact_ref("technical_report"),
         "The dataset is not the complete Iranian housing stock or transaction universe."),
        ("25 - Shared checklist", "n is reported for all professor-facing analytical outputs", "REVIEW",
         "Most canonical outputs carry N; final notebook/report presentation still needs a display-level completeness check.",
         _combine_evidence("price_driver_diagnostics", "seller_summary", "text_summary", "segment_profile"),
         "This is a presentation check, not a reason to recompute any model."),
        ("25 - Shared checklist", "Missing meaning is documented for important fields",
         "PASS" if _artifact_exists("data_dictionary") else "REVIEW",
         "Canonical policy preserves NULL/non-applicability and uses task-specific imputation rather than blanket zero/False filling.",
         _artifact_ref("data_dictionary"),
         "If the canonical dictionary path is not found, keep REVIEW until its actual final path is registered."),
        ("25 - Shared checklist", "Duplicate handling is documented",
         _status_from_evidence("duplicate_summary"),
         "Conservative duplicate/entity-resolution logic is upstream and lineage-preserving.",
         _artifact_ref("duplicate_summary"),
         "Medium candidates and cross-month repeats are not automatically deleted."),
        ("25 - Shared checklist", "Outlier policy is documented",
         _status_from_evidence("outlier_summary", "outlier_sensitivity"),
         "Group-aware canonical outlier flags plus sensitivity alternatives are documented.",
         _combine_evidence("outlier_summary", "outlier_sensitivity"),
         "Outliers are flags, not automatic row deletions."),
        ("25 - Shared checklist", "Mean versus Median is chosen deliberately", "PASS",
         "Skewed asking-price summaries use robust median/IQR where appropriate.",
         _artifact_ref("monthly_price"),
         "Do not run a mean only to satisfy a generic checklist when median is the declared estimand."),
        ("25 - Shared checklist", "Correlation is not confused with causation", "PASS",
         "Price-driver, seller and text outputs use explicit observational/predictive semantics.",
         _combine_evidence("price_driver_manifest", "seller_manifest", "text_manifest"),
         "Association and predictive importance are not causal effects."),
        ("25 - Shared checklist", "Main confounders / controls are explicit", "PASS",
         "Location, property structure/type, time and other observed controls are handled in model specifications.",
         _combine_evidence("price_driver_manifest", "seller_manifest"),
         "Residual confounding can remain even after adjustment."),
        ("25 - Shared checklist", "Statistical test matches data and estimand", "PASS",
         "Seller/text modules document the tests/contrasts used for controlled observational comparisons.",
         _combine_evidence("seller_summary", "text_summary"),
         "Descriptive outputs do not require a hypothesis test merely because a generic checklist mentions testing."),
        ("25 - Shared checklist", "Effect size and uncertainty are reported where inferentially applicable", "PASS",
         "Adjusted seller/text contrasts include uncertainty; descriptive/predictive outputs retain their own diagnostics.",
         _combine_evidence("seller_summary", "text_summary"),
         "Do not create meaningless CI bars when the accepted method does not define useful inferential uncertainty."),
        ("25 - Shared checklist", "Multiple testing is controlled where applicable",
         _status_from_evidence("text_summary"),
         "Validated text signals use Benjamini-Hochberg FDR across the tested family.",
         _artifact_ref("text_summary"),
         "FDR is task-specific; it is not applied mechanically to unrelated descriptive outputs."),
        ("25 - Shared checklist", "Data leakage is explicitly guarded",
         _status_from_evidence("price_driver_manifest", "text_manifest"),
         "Held-out evaluation and the control-only text design prevent keyword leakage into the price-control model.",
         _combine_evidence("price_driver_manifest", "text_manifest"),
         "Final reporting must not refit models on held-out evidence."),
        ("25 - Shared checklist", "Sensitivity / robustness is defined", "PASS",
         "Canonical sensitivity evidence is consolidated in robustness_sensitivity_matrix.csv.",
         relative_to_project(ROBUSTNESS_PATH),
         "Only executed project-specific checks are claimed."),
        ("25 - Shared checklist", "Limitations and claim boundaries are documented",
         "PASS" if _artifact_exists("technical_report") else "REVIEW",
         "Final reporting carries asking-price, currency, platform, model, spatial, segmentation and temporal limitations.",
         _artifact_ref("technical_report"),
         "Do not describe limitations as preregistered unless they actually were preregistered."),

        ("28 - Housing checklist", "Currency is specified", currency_status,
         currency_note,
         _artifact_ref("quality_gate"),
         "Operational Toman is usable with the explicit source-unconfirmed caveat."),
        ("28 - Housing checklist", "Sale / rent / deposit regimes are separated",
         _status_from_evidence("gold_price_regime"),
         "The complete price-regime taxonomy prevents incompatible monetary structures from being mixed.",
         _artifact_ref("gold_price_regime"),
         "Temporary rent, service and unknown/incomplete states remain semantically separate."),
        ("28 - Housing checklist", "Duplicate is checked before supply analysis",
         _status_from_evidence("duplicate_summary", "duplicate_supply_impact"),
         "Supply eligibility consumes the canonical conservative duplicate policy.",
         _combine_evidence("duplicate_summary", "duplicate_supply_impact"),
         "Cross-month repeats may represent continued platform activity and remain retained."),
        ("28 - Housing checklist", "Outliers are assessed within appropriate groups",
         _status_from_evidence("outlier_summary"),
         "Canonical outlier thresholds are context/group-aware with documented fallbacks.",
         _artifact_ref("outlier_summary"),
         "No arbitrary global deletion rule replaces the canonical policy."),
        ("28 - Housing checklist", "PSM has a valid denominator",
         _status_from_evidence("final_metrics"),
         "Sale asking PSM is published only for eligible records with a valid compatible area denominator.",
         _artifact_ref("final_metrics"),
         "Outside the valid population, final PSM remains non-analytical/NULL."),
        ("28 - Housing checklist", "Minimum neighborhood sample size / reliability gate is defined",
         _status_from_evidence("market_temperature_manifest"),
         "Neighborhood-level outputs use task-specific minimum-N/reliability requirements.",
         _artifact_ref("market_temperature_manifest"),
         "Minimum N is a parameter/gate, not a business dimension."),
        ("28 - Housing checklist", "Median and IQR are reported",
         _status_from_evidence("monthly_price"),
         "Monthly apartment-sale asking PSM includes median and P25-P75 dispersion.",
         _artifact_ref("monthly_price"),
         "IQR is cross-sectional dispersion, not uncertainty around the median."),
        ("28 - Housing checklist", "Property characteristics are evaluated with location/area controls",
         _status_from_evidence("price_driver_manifest"),
         "Price-driver modeling separates structural controls from property-characteristic contrasts.",
         _artifact_ref("price_driver_manifest"),
         "Adjusted contrasts are observational and model-implied."),
        ("28 - Housing checklist", "Rent/deposit conversion has sensitivity scenarios",
         _status_from_evidence("final_metrics", "settings"),
         "Low/base/high conversion scenarios are part of the frozen rental-equivalent contract.",
         _combine_evidence("final_metrics", "settings"),
         "Scenarios are project assumptions, not verified official conversion rates."),
        ("28 - Housing checklist", "Asking price is not confused with transaction price",
         _status_from_evidence("quality_gate"),
         "The canonical price-observation contract is asking_price throughout downstream analysis.",
         _artifact_ref("quality_gate"),
         "No realized transaction-price claim is permitted."),
    ]

    return pd.DataFrame(
        rows,
        columns=[
            "source_checklist",
            "requirement",
            "status",
            "project_implementation",
            "canonical_evidence",
            "claim_boundary_or_action",
        ],
    )


def build_design_summary() -> pd.DataFrame:
    seller = _seller_context()
    text = _text_context()
    seg = _segmentation_context()

    rows = [
        ("Population",
         "Platform real-estate listings meeting each task's explicit eligibility, regime and time contract.",
         _combine_evidence("final_metrics", "segment_manifest")),
        ("Sample",
         "Observed platform listings; task-specific n is read from accepted analytical artifacts.",
         _combine_evidence("price_driver_diagnostics", "seller_summary", "text_summary")),
        ("Unit of Analysis",
         "Primarily listing-level for cleaning/modeling; selected market summaries use city-, neighborhood- or month-level aggregate grains.",
         _combine_evidence("segment_manifest", "market_temperature_manifest")),
        ("Main Biases",
         "Platform/coverage selection, duplicate/relisting bias, asking-price measurement bias and residual spatial/compositional confounding.",
         _artifact_ref("technical_report")),
        ("Five Core Variables",
         "Asking PSM; area; rooms; building age; location (City + Neighborhood).",
         _combine_evidence("price_driver_diagnostics", "price_driver_importance")),
        ("Missing Policy",
         "Preserve NULL/non-applicability in Silver; avoid blanket zero/False filling; use task-specific imputation only inside the relevant model/analysis.",
         _artifact_ref("data_dictionary")),
        ("Outlier Policy",
         "Flag-only group-aware 3x IQR canonical rule; 2x IQR and log-MAD retained as sensitivity evidence.",
         _combine_evidence("outlier_summary", "outlier_sensitivity")),
        ("Descriptive Statistics",
         "Median/IQR for skewed asking-price summaries; counts/shares and task-specific trend/dispersion statistics elsewhere.",
         _artifact_ref("monthly_price")),
        ("Hypothesis 1 - Seller Type",
         "H0: adjusted agency-minus-personal asking-PSM contrast = 0; H1: adjusted contrast != 0. "
         f"Accepted estimates: raw={_fmt_pct(seller['raw'])}, adjusted={_fmt_pct(seller['adjusted'])}, strata={_fmt_pct(seller['stratified'])}.",
         _artifact_ref("seller_summary")),
        ("Hypothesis 2 - Text Signals",
         "For each precision-validated keyword family: H0 adjusted residual contrast = 0; H1 contrast != 0, with BH-FDR controlling the family. "
         f"Accepted family summary: tested={text['tested']}, FDR-significant={text['fdr_significant']}.",
         _combine_evidence("text_summary", "text_precision")),
        ("Effect Size / Uncertainty",
         "Report accepted seller/text adjusted percentage contrasts with uncertainty; use predictive diagnostics for AVM/importance outputs rather than causal-effect language.",
         _combine_evidence("seller_summary", "text_summary", "price_driver_diagnostics")),
        ("Sensitivity Checks",
         "Outlier threshold/method sensitivity; rent/deposit scenarios; duplicate raw-vs-deduplicated activity; Market Temperature sensitivity; seller and segmentation robustness.",
         relative_to_project(ROBUSTNESS_PATH)),
        ("Libraries",
         "Polars/Pandas for data work, NumPy/SciPy for statistics, scikit-learn for preprocessing/modeling/clustering, Matplotlib for technical figures.",
         "Project source modules / environment"),
        ("Segmentation Release",
         f"Accepted primary method={seg['primary_method']}; final market types are descriptive and method-aware.",
         _combine_evidence("segment_profile", "segment_manifest", "cluster_diagnostics")),
        ("Forbidden Claim 1",
         "Do not call listing asking prices realized transaction prices.",
         _artifact_ref("quality_gate")),
        ("Forbidden Claim 2",
         "Do not call platform listing activity physical housing inventory, liquidity or absorption.",
         _artifact_ref("market_temperature_manifest")),
        ("Forbidden Claim 3",
         "Do not call adjusted price-driver, seller-type or text associations causal effects.",
         _combine_evidence("price_driver_manifest", "seller_manifest", "text_manifest")),
        ("Forbidden Claim 4",
         "Do not call descriptive market types stable clusters, socioeconomic classes or buyer personas unless a future accepted artifact supports that wording.",
         _combine_evidence("segment_profile", "segment_manifest")),
    ]
    return pd.DataFrame(rows, columns=["section", "project_summary", "canonical_evidence"])


def _referenced_paths(frame: pd.DataFrame) -> list[str]:
    if "canonical_evidence" not in frame.columns:
        return []
    values: list[str] = []
    for raw in frame["canonical_evidence"].fillna("").astype(str):
        for item in raw.split(";"):
            item = item.strip()
            if (
                not item
                or item.startswith("MISSING:")
                or item == "Project source modules / environment"
            ):
                continue
            values.append(item)
    return values


def build_checks(
    method_map: pd.DataFrame,
    robustness: pd.DataFrame,
    readiness: pd.DataFrame,
    design: pd.DataFrame,
) -> list[Check]:
    referenced = set()
    for frame in (method_map, robustness, readiness, design):
        referenced.update(_referenced_paths(frame))

    missing_referenced = [
        path for path in sorted(referenced)
        if not (PROJECT_ROOT / path).exists()
    ]

    readiness_failures = int((readiness["status"] == "FAIL").sum())
    currency_rows = readiness.loc[
        (readiness["source_checklist"] == "28 - Housing checklist")
        & (readiness["requirement"] == "Currency is specified"),
        "status",
    ]
    currency_status = currency_rows.iloc[0] if not currency_rows.empty else "missing"
    currency_ok = currency_status in {"PASS", "REVIEW"}

    seg = _segmentation_context()
    kmeans_rows = method_map.loc[
        method_map["concept"] == "MiniBatchKMeans / K-means",
        "implementation_status",
    ]
    clustering_claim_guard = True
    if "rule" in seg["primary_method"].lower() and not kmeans_rows.empty:
        clustering_claim_guard = kmeans_rows.iloc[0] != "RELEASED_AS_PRIMARY"

    return [
        make_check(
            "statistical_method_map_nonempty",
            "final_reporting",
            len(method_map),
            ">0",
            len(method_map) > 0,
        ),
        make_check(
            "robustness_matrix_nonempty",
            "final_reporting",
            len(robustness),
            ">0",
            len(robustness) > 0,
        ),
        make_check(
            "readiness_matrix_nonempty",
            "final_reporting",
            len(readiness),
            ">0",
            len(readiness) > 0,
        ),
        make_check(
            "statistical_design_summary_nonempty",
            "final_reporting",
            len(design),
            ">0",
            len(design) > 0,
        ),
        make_check(
            "readiness_has_no_fail",
            "final_reporting",
            readiness_failures,
            0,
            readiness_failures == 0,
            notes=(
                "REVIEW is allowed for documented non-critical limitations such as "
                "source-unconfirmed currency; FAIL means required evidence is missing."
            ),
        ),
        make_check(
            "currency_review_preserved",
            "semantic_contract",
            currency_status,
            "PASS or REVIEW",
            currency_ok,
            notes=(
                "Do not force currency to PASS merely for presentation; "
                "toman_assumed_unconfirmed may remain a documented REVIEW state."
            ),
        ),
        make_check(
            "segmentation_claim_guard",
            "semantic_contract",
            seg["primary_method"],
            "final wording matches accepted segmentation method",
            clustering_claim_guard,
            notes=(
                "A rule-based/descriptive final release must not be relabeled "
                "as K-means clustering."
            ),
        ),
        make_check(
            "referenced_evidence_paths_resolve",
            "lineage",
            len(missing_referenced),
            0,
            len(missing_referenced) == 0,
            critical=False,
            review_on_fail=True,
            notes="Missing references: "
            + ("; ".join(missing_referenced) if missing_referenced else "none"),
        ),
        make_check(
            "documentation_only_contract",
            "method",
            "no model fitting or metric recomputation",
            "documentation-only",
            True,
            notes=(
                "This module reads accepted artifacts and builds governance tables only; "
                "it does not modify Silver, Gold, models, dashboard logic or analytical estimates."
            ),
        ),
    ]


def run(*, strict: bool = False) -> dict[str, Path]:
    started = time.perf_counter()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    QA_DIR.mkdir(parents=True, exist_ok=True)

    method_map = build_statistical_method_map()
    robustness = build_robustness_matrix()
    readiness = build_readiness_matrix()
    design = build_design_summary()

    atomic_write_csv(pl.from_pandas(method_map), METHOD_MAP_PATH)
    atomic_write_csv(pl.from_pandas(robustness), ROBUSTNESS_PATH)
    atomic_write_csv(pl.from_pandas(readiness), READINESS_PATH)
    atomic_write_csv(pl.from_pandas(design), DESIGN_SUMMARY_PATH)

    checks = build_checks(method_map, robustness, readiness, design)
    atomic_write_csv(checks_frame(checks), CHECKS_PATH)
    status = summarize_checks(checks)

    artifact_inventory: dict[str, Any] = {}
    for name, candidates in ARTIFACTS.items():
        resolved = _artifact_path(name)
        artifact_inventory[name] = {
            "resolved_path": relative_to_project(resolved) if resolved is not None else None,
            "exists": resolved is not None,
            "candidates": [relative_to_project(path) for path in candidates],
        }

    manifest = {
        "version": VERSION,
        "status": status,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "contracts": {
            "documentation_only": True,
            "refits_models": False,
            "recomputes_analytical_estimates": False,
            "writes_silver": False,
            "writes_gold": False,
            "writes_dashboard": False,
            "creates_new_professor_facing_report": False,
            "intended_consumers": [
                "notebooks/final_analysis.ipynb",
                "src/final_reporting/build_technical_report.py",
                "src/final_reporting/build_executive_summary.py (summary-level only, if needed)",
            ],
        },
        "outputs": {
            "statistical_method_map": relative_to_project(METHOD_MAP_PATH),
            "robustness_sensitivity_matrix": relative_to_project(ROBUSTNESS_PATH),
            "statistical_readiness_matrix": relative_to_project(READINESS_PATH),
            "statistical_design_summary": relative_to_project(DESIGN_SUMMARY_PATH),
            "checks": relative_to_project(CHECKS_PATH),
        },
        "artifact_inventory": artifact_inventory,
        "interpretation_contract": {
            "price": "asking price, not transaction price",
            "currency": "operational Toman; source status may remain toman_assumed_unconfirmed",
            "listing_activity": "platform listing activity; not physical inventory, liquidity or absorption",
            "model_outputs": "observational/predictive, not causal",
            "segmentation": (
                "method-aware descriptive market types unless accepted clustering "
                "gates support stronger wording"
            ),
        },
    }
    atomic_write_json(manifest, MANIFEST_PATH)

    if strict and (
        str(status.get("overall_status", "")).upper() == "FAIL"
        or int(status.get("critical_failures", 0) or 0) > 0
    ):
        raise RuntimeError(
            "Statistical-governance build failed strict validation. "
            f"See {relative_to_project(CHECKS_PATH)}."
        )

    return {
        "method_map": METHOD_MAP_PATH,
        "robustness": ROBUSTNESS_PATH,
        "readiness": READINESS_PATH,
        "design_summary": DESIGN_SUMMARY_PATH,
        "checks": CHECKS_PATH,
        "manifest": MANIFEST_PATH,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build final statistical-governance documentation tables from accepted IHMI "
            "artifacts without refitting models or changing analytical results."
        )
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Raise when critical governance/evidence checks fail.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run(strict=args.strict)
    print("FINAL STATISTICAL GOVERNANCE COMPLETED")
    for name, path in outputs.items():
        print(f"{name}: {relative_to_project(path)}")


if __name__ == "__main__":
    main()
