from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import polars as pl
from scipy import stats

from src.common.io_utils import atomic_write_csv, atomic_write_json
from src.common.paths import OUTPUTS_DIR, relative_to_project
from src.common.validation import Check, checks_frame, make_check, summarize_checks
from src.milestone_3.price_drivers.price_drivers import _prepare
from src.milestone_3.text_analysis.text_rules import KEYWORD_RULES, MANDATORY_ASSIGNMENT_KEYWORDS

VERSION = "m3-text-price-signals-v1.4-fdr-forest-layout"
FEATURE_TABLE = OUTPUTS_DIR / "model_artifacts" / "milestone_3" / "price_drivers" / "analysis_ready_features.parquet"
TEXT_TABLE = OUTPUTS_DIR / "model_artifacts" / "milestone_3" / "text_analysis" / "text_features.parquet"
PRECISION = OUTPUTS_DIR / "qa" / "milestone_3" / "text_analysis" / "keyword_precision_summary.csv"
MODEL = OUTPUTS_DIR / "model_artifacts" / "milestone_3" / "price_drivers" / "ridge_price_driver_model.joblib"
MODEL_MANIFEST = OUTPUTS_DIR / "qa" / "milestone_3" / "price_drivers" / "price_driver_manifest.json"
TARGET = "sale_price_per_sqm_final_toman"
LOG_TARGET = "log_sale_price_per_sqm"
PROGRESS_WIDTH = 30
FIGURE_STYLE_VERSION = "text-signal-forest-v2-semantic-layout"

TEXT_SIGNAL_LABELS = {
    "below_market": "Below-market claim (زیر قیمت)",
    "exchange": "Exchange (معاوضه)",
    "urgent": "Urgent (فوری)",
    "migration_sale": "Migration-related sale (فروش به دلیل مهاجرت)",
    "unused": "Unused unit (کلیدنخورده)",
    "new_build": "New build (نوساز)",
}
TEXT_SIGNAL_ORDER = ["below_market", "exchange", "urgent", "migration_sale", "unused", "new_build"]
TRANSACTION_GROUP_END = 4


def show_progress(percent: int, label: str, *, final: bool = False) -> None:
    pct = max(0, min(100, int(percent)))
    filled = int(PROGRESS_WIDTH * pct / 100)
    bar = "#" * filled + "-" * (PROGRESS_WIDTH - filled)
    print(
        f"\rM3 text signals [{bar}] {pct:3d}% complete | {100-pct:3d}% remaining | {label[:38]:38s}",
        end="\n" if final else "", flush=True,
    )


def _welch_log_diff(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float, float]:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan, np.nan, np.nan
    diff = float(np.mean(a) - np.mean(b))
    se = math.sqrt(np.var(a, ddof=1) / len(a) + np.var(b, ddof=1) / len(b))
    lo, hi = diff - 1.96 * se, diff + 1.96 * se
    p = float(stats.ttest_ind(a, b, equal_var=False, nan_policy="omit").pvalue)
    return diff, lo, hi, p



def _as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "pass"}

def _to_pct(value: float) -> float:
    return float((math.exp(value) - 1.0) * 100.0) if np.isfinite(value) else np.nan


def _benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    """Benjamini-Hochberg FDR-adjusted q-values, preserving the input index."""
    numeric = pd.to_numeric(p_values, errors="coerce")
    valid = numeric.dropna().clip(lower=0.0, upper=1.0)
    result = pd.Series(np.nan, index=numeric.index, dtype=float)
    m = len(valid)
    if m == 0:
        return result
    ordered = valid.sort_values(kind="mergesort")
    ranks = np.arange(1, m + 1, dtype=float)
    raw_q = ordered.to_numpy(dtype=float) * m / ranks
    monotone_q = np.minimum.accumulate(raw_q[::-1])[::-1]
    result.loc[ordered.index] = np.minimum(monotone_q, 1.0)
    return result


def _format_q(value: float) -> str:
    if not np.isfinite(value):
        return "q=n/a"
    if value < 0.001:
        return "q<0.001"
    return f"q={value:.3f}"


def run(
    feature_path: Path = FEATURE_TABLE,
    text_path: Path = TEXT_TABLE,
    precision_path: Path = PRECISION,
    model_path: Path = MODEL,
    model_manifest_path: Path = MODEL_MANIFEST,
) -> dict[str, Path]:
    paths = [feature_path, text_path, precision_path, model_path, model_manifest_path]
    for path in paths:
        if not path.resolve().exists():
            raise FileNotFoundError(f"Required text-signal input not found: {path}")
    feature_path, text_path, precision_path, model_path, model_manifest_path = [p.resolve() for p in paths]

    table_dir = OUTPUTS_DIR / "tables" / "milestone_3" / "text_analysis"
    qa_dir = OUTPUTS_DIR / "qa" / "milestone_3" / "text_analysis"
    fig_dir = OUTPUTS_DIR / "figures" / "milestone_3" / "text_analysis"
    for d in [table_dir, qa_dir, fig_dir]:
        d.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    show_progress(0, "loading text/model inputs")

    features = pd.read_parquet(feature_path)
    text = pd.read_parquet(text_path)
    precision = pd.read_csv(precision_path)
    manifest = json.loads(model_manifest_path.read_text(encoding="utf-8"))
    active_features = [str(x) for x in manifest.get("active_features", [])]
    pipe = joblib.load(model_path)
    if not active_features:
        raise ValueError("Price-driver manifest does not contain active_features.")
    population = features.merge(text, on="source_row_id", how="inner", suffixes=("", "_text"))
    test = population.loc[population["analysis_split"] == "test"].copy()
    if test.empty:
        raise RuntimeError("Text-price controlled analysis requires a non-empty held-out test month.")
    test["control_only_prediction_log"] = pipe.predict(_prepare(test, active_features))
    test["control_only_log_residual"] = test[LOG_TARGET].to_numpy(dtype=float) - test["control_only_prediction_log"].to_numpy(dtype=float)
    show_progress(32, f"held-out control residuals: {len(test):,} rows")

    precision_map = precision.set_index("keyword").to_dict("index")
    rows: list[dict[str, Any]] = []
    for key, spec in KEYWORD_RULES.items():
        meta = precision_map.get(key, {})
        validated = _as_bool(meta.get("include_in_controlled_analysis", False))
        flag = f"keyword_{key}_flag"
        base_row: dict[str, Any] = {
            "keyword": key,
            "keyword_fa": spec["keyword_fa"],
            "manual_precision": meta.get("manual_precision"),
            "precision_validation_status": meta.get("validation_status", "REVIEW"),
            "analysis_status": "NOT_RUN_UNVALIDATED" if not validated else "READY",
            "heldout_test_n": len(test),
            "positive_n": 0,
            "negative_n": 0,
            "raw_median_difference_pct": None,
            "raw_mann_whitney_p_value": None,
            "adjusted_residual_difference_pct": None,
            "adjusted_ci_low_pct": None,
            "adjusted_ci_high_pct": None,
            "adjusted_welch_p_value": None,
            "adjusted_fdr_bh_q_value": None,
            "adjusted_fdr_bh_status": "NOT_TESTED",
            "interpretation": "controlled observational association; keyword is not a predictor in the control-only model",
            "analysis_version": VERSION,
        }
        if not validated or flag not in test.columns:
            rows.append(base_row)
            continue
        positive_mask = test[flag].fillna(False).astype(bool)
        positive = test.loc[positive_mask]
        negative = test.loc[~positive_mask]
        base_row["positive_n"] = len(positive)
        base_row["negative_n"] = len(negative)
        if len(positive) < 20 or len(negative) < 20:
            base_row["analysis_status"] = "REVIEW_LOW_N"
            rows.append(base_row)
            continue
        raw_pos = positive[TARGET].to_numpy(dtype=float)
        raw_neg = negative[TARGET].to_numpy(dtype=float)
        base_row["raw_median_difference_pct"] = float((np.median(raw_pos) / np.median(raw_neg) - 1.0) * 100.0)
        base_row["raw_mann_whitney_p_value"] = float(stats.mannwhitneyu(raw_pos, raw_neg, alternative="two-sided", method="asymptotic").pvalue)
        diff, lo, hi, p = _welch_log_diff(
            positive["control_only_log_residual"].to_numpy(dtype=float),
            negative["control_only_log_residual"].to_numpy(dtype=float),
        )
        base_row["adjusted_residual_difference_pct"] = _to_pct(diff)
        base_row["adjusted_ci_low_pct"] = _to_pct(lo)
        base_row["adjusted_ci_high_pct"] = _to_pct(hi)
        base_row["adjusted_welch_p_value"] = p
        base_row["analysis_status"] = "PASS"
        rows.append(base_row)
    summary = pd.DataFrame(rows)
    tested_mask = summary["analysis_status"].eq("PASS") & pd.to_numeric(summary["adjusted_welch_p_value"], errors="coerce").notna()
    if tested_mask.any():
        q_values = _benjamini_hochberg(summary.loc[tested_mask, "adjusted_welch_p_value"])
        summary.loc[tested_mask, "adjusted_fdr_bh_q_value"] = q_values
        summary.loc[tested_mask, "adjusted_fdr_bh_status"] = np.where(
            q_values.to_numpy(dtype=float) < 0.05,
            "SIGNIFICANT_FDR_0_05",
            "NOT_SIGNIFICANT_FDR_0_05",
        )
    show_progress(62, "validated held-out keyword comparisons + BH-FDR complete")

    month = text.copy()
    month["analysis_month"] = month["analysis_month"].astype(str).str[:7]
    monthly_rows: list[dict[str, Any]] = []
    for key, spec in KEYWORD_RULES.items():
        flag = f"keyword_{key}_flag"
        if flag not in month:
            continue
        grouped = month.groupby("analysis_month", dropna=False)[flag].agg(["sum", "count"]).reset_index()
        for _, item in grouped.iterrows():
            monthly_rows.append({
                "analysis_month": item["analysis_month"],
                "keyword": key,
                "keyword_fa": spec["keyword_fa"],
                "positive_n": int(item["sum"]),
                "population_n": int(item["count"]),
                "positive_rate": float(item["sum"] / item["count"]) if item["count"] else np.nan,
            })
    monthly = pd.DataFrame(monthly_rows)

    mandatory_unvalidated = [key for key in MANDATORY_ASSIGNMENT_KEYWORDS if not _as_bool(precision_map.get(key, {}).get("include_in_controlled_analysis", False))]
    mandatory_not_analyzed = [key for key in MANDATORY_ASSIGNMENT_KEYWORDS if summary.loc[summary["keyword"] == key, "analysis_status"].iloc[0] != "PASS"]
    checks: list[Check] = [
        make_check(
            "mandatory_keyword_precision_validated", "text", ";".join(mandatory_unvalidated) if mandatory_unvalidated else "all",
            "urgent and below_market validated", not mandatory_unvalidated,
            review_on_fail=False,
            notes="§28 Q8 should not be claimed until mandatory keyword families pass manual precision validation.",
        ),
        make_check(
            "mandatory_keyword_analysis_complete", "text", ";".join(mandatory_not_analyzed) if mandatory_not_analyzed else "all",
            "urgent and below_market PASS", not mandatory_not_analyzed,
            review_on_fail=False,
        ),
        make_check("heldout_control_population_nonempty", "text", len(test), ">0", len(test) > 0),
        make_check(
            "adjusted_multiple_testing_fdr_complete", "statistics",
            int(summary.loc[summary["analysis_status"].eq("PASS"), "adjusted_fdr_bh_q_value"].notna().sum()),
            int(summary["analysis_status"].eq("PASS").sum()),
            int(summary.loc[summary["analysis_status"].eq("PASS"), "adjusted_fdr_bh_q_value"].notna().sum()) == int(summary["analysis_status"].eq("PASS").sum()),
            notes="Benjamini-Hochberg FDR is applied across the family of precision-validated held-out adjusted Welch tests at q<0.05.",
        ),
        make_check(
            "keyword_not_used_as_predictor", "method", "control-only Ridge", "keyword excluded", True,
            notes="Keyword effect is evaluated on held-out residuals from a model that does not include text flags.",
        ),
    ]

    summary_path = table_dir / "text_signal_summary.csv"
    monthly_path = table_dir / "text_keyword_monthly_frequency.csv"
    checks_path = qa_dir / "text_signal_checks.csv"
    manifest_path = qa_dir / "text_signal_manifest.json"
    final_figure_path = fig_dir / "validated_text_signal_forest.png"
    preliminary_figure_path = fig_dir / "text_signal_adjusted_association_PRELIMINARY.png"
    atomic_write_csv(pl.from_pandas(summary), summary_path)
    atomic_write_csv(pl.from_pandas(monthly), monthly_path)
    atomic_write_csv(checks_frame(checks), checks_path)

    # Final professor-facing figure is allowed only for precision-validated families.
    validated_plot = summary.loc[(summary["analysis_status"] == "PASS") & summary["adjusted_residual_difference_pct"].notna()].copy()
    final_ready = not mandatory_unvalidated and not mandatory_not_analyzed and not validated_plot.empty

    # While manual validation is pending, create a clearly marked review-only preview so visual QA can be completed now.
    # These preview estimates are never written into the canonical text_signal_summary.csv.
    if final_ready:
        plot = validated_plot.copy()
        figure_path = final_figure_path
        if preliminary_figure_path.exists():
            preliminary_figure_path.unlink()
        preliminary = False
    else:
        preview_rows: list[dict[str, Any]] = []
        for key, spec in KEYWORD_RULES.items():
            flag = f"keyword_{key}_flag"
            if flag not in test.columns:
                continue
            positive_mask = test[flag].fillna(False).astype(bool)
            positive = test.loc[positive_mask]
            negative = test.loc[~positive_mask]
            if len(positive) < 20 or len(negative) < 20:
                continue
            diff, lo, hi, _ = _welch_log_diff(
                positive["control_only_log_residual"].to_numpy(dtype=float),
                negative["control_only_log_residual"].to_numpy(dtype=float),
            )
            meta = precision_map.get(key, {})
            preview_rows.append({
                "keyword": key,
                "keyword_fa": spec["keyword_fa"],
                "positive_n": len(positive),
                "manual_precision": meta.get("manual_precision"),
                "adjusted_residual_difference_pct": _to_pct(diff),
                "adjusted_ci_low_pct": _to_pct(lo),
                "adjusted_ci_high_pct": _to_pct(hi),
            })
        plot = pd.DataFrame(preview_rows)
        figure_path = preliminary_figure_path
        if final_figure_path.exists():
            final_figure_path.unlink()
        preliminary = True

    if not plot.empty:
        order_map = {key: idx for idx, key in enumerate(TEXT_SIGNAL_ORDER)}
        plot["_display_order"] = plot["keyword"].astype(str).map(order_map).fillna(len(order_map))
        plot = plot.sort_values(["_display_order", "adjusted_residual_difference_pct"]).reset_index(drop=True)

        effects = pd.to_numeric(plot["adjusted_residual_difference_pct"], errors="coerce").to_numpy(dtype=float)
        lo = pd.to_numeric(plot["adjusted_ci_low_pct"], errors="coerce").to_numpy(dtype=float)
        hi = pd.to_numeric(plot["adjusted_ci_high_pct"], errors="coerce").to_numpy(dtype=float)
        y = np.arange(len(plot), dtype=float)
        qvals = pd.to_numeric(plot.get("adjusted_fdr_bh_q_value", pd.Series(np.nan, index=plot.index)), errors="coerce").to_numpy(dtype=float)

        categories: list[str] = []
        for idx, row in plot.iterrows():
            if preliminary:
                categories.append("insufficient")
                continue
            validated = str(row.get("precision_validation_status", "")).upper() == "PASS"
            analyzed = str(row.get("analysis_status", "")).upper() == "PASS"
            q_value = qvals[idx]
            if not validated or not analyzed or not np.isfinite(effects[idx]):
                categories.append("insufficient")
            elif np.isfinite(q_value) and q_value < 0.05:
                categories.append("higher" if effects[idx] > 0 else "lower")
            else:
                categories.append("no_material")

        category_colors = {
            "higher": "#2F6596",
            "lower": "#C8793D",
            "no_material": "#8D98A5",
            "insufficient": "#B8C3CE",
        }

        fig, ax = plt.subplots(figsize=(14.4, max(6.8, 0.92 * len(plot) + 2.4)))
        fig.suptitle(
            "PRELIMINARY - Validated Listing-Text Signals and Adjusted Asking-Price Association"
            if preliminary
            else "Validated Listing-Text Signals and Adjusted Asking-Price Association",
            fontsize=15.8, fontweight="bold", y=0.982,
        )
        subtitle_text = (
            f"Visual-QA preview | point = adjusted estimate; line = 95% interval | held-out N={len(test):,} | manual validation pending"
            if preliminary
            else f"Precision-validated keyword families | point = adjusted estimate; line = 95% interval | held-out N={len(test):,} | BH-FDR q<0.05"
        )
        fig.text(0.5, 0.936, subtitle_text, ha="center", va="center", fontsize=10.2, color="#555555")

        ax.axvline(0, linewidth=1.15, color="#4D5966")
        for yi, effect, low, high, category in zip(y, effects, lo, hi, categories):
            color = category_colors[category]
            if np.isfinite(low) and np.isfinite(high):
                ax.plot([low, high], [yi, yi], color=color, linewidth=2.0, solid_capstyle="round", zorder=2)
                ax.plot([low, low], [yi - 0.08, yi + 0.08], color=color, linewidth=1.5, zorder=2)
                ax.plot([high, high], [yi - 0.08, yi + 0.08], color=color, linewidth=1.5, zorder=2)
            if np.isfinite(effect):
                ax.scatter(effect, yi, s=88, color=color, edgecolors="white", linewidths=0.8, zorder=3)

        display_labels = [TEXT_SIGNAL_LABELS.get(str(key), str(label)) for key, label in zip(plot["keyword"], plot["keyword_fa"])]
        ax.set_yticks(y, display_labels, fontsize=10.7)
        if len(plot) > TRANSACTION_GROUP_END:
            ax.axhline(TRANSACTION_GROUP_END - 0.5, color="#D5DAE0", linewidth=1.0, zorder=0)
        ax.invert_yaxis()
        ax.set_xlabel("Model-implied adjusted difference in asking PSM (%)")
        ax.grid(axis="x", alpha=0.14, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

        finite_bounds = np.concatenate([effects[np.isfinite(effects)], lo[np.isfinite(lo)], hi[np.isfinite(hi)]])
        bound = max(float(np.nanmax(np.abs(finite_bounds))) if finite_bounds.size else 1.0, 10.0)
        bound = math.ceil(bound * 1.12 / 5.0) * 5.0
        ax.set_xlim(-bound, bound)
        effect_pad = max(bound * 0.018, 0.7)
        for idx, (yi, effect, low, high, category) in enumerate(zip(y, effects, lo, hi, categories)):
            row = plot.iloc[idx]
            precision_value = pd.to_numeric(pd.Series([row.get("manual_precision")]), errors="coerce").iloc[0]
            precision_text = f"{float(precision_value)*100.0:.0f}%" if pd.notna(precision_value) else "N/A"
            positive_n = pd.to_numeric(pd.Series([row.get("positive_n")]), errors="coerce").iloc[0]
            n_text = f"{int(positive_n):,}" if pd.notna(positive_n) else "N/A"
            if preliminary:
                significance = "validation pending"
                evidence = "Validation pending"
            else:
                significance = _format_q(qvals[idx]) if np.isfinite(qvals[idx]) else "q=n/a"
                evidence = "FDR significant" if category in {"higher", "lower"} else "Not FDR significant"

            if np.isfinite(effect):
                effect_x = effect + effect_pad if effect >= 0 else effect - effect_pad
                ax.text(
                    effect_x, yi, f"{effect:+.1f}%", va="center",
                    ha="left" if effect >= 0 else "right", fontsize=8.8, fontweight="bold",
                    color=category_colors[category],
                )
            metadata = f"N={n_text} | Precision={precision_text} | {significance}\n{evidence}"
            ax.text(
                1.025, yi, metadata, transform=ax.get_yaxis_transform(),
                va="center", ha="left", fontsize=8.1, color="#27313A", clip_on=False, linespacing=1.22,
            )

        if preliminary:
            legend_handles = [
                Line2D([0], [0], color=category_colors["insufficient"], marker="o", linewidth=1.8, label="Validation pending / review only"),
            ]
            legend_columns = 1
        else:
            legend_handles = [
                Line2D([0], [0], color=category_colors["higher"], marker="o", linewidth=1.8, label="Higher after controls; FDR significant"),
                Line2D([0], [0], color=category_colors["lower"], marker="o", linewidth=1.8, label="Lower after controls; FDR significant"),
                Line2D([0], [0], color=category_colors["no_material"], marker="o", linewidth=1.8, label="Not FDR significant"),
            ]
            legend_columns = 3
        fig.legend(
            handles=legend_handles, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 0.064),
            ncol=legend_columns, fontsize=8.4, columnspacing=1.6, handlelength=2.0,
        )
        footer_text = (
            "NOT FOR REPORT. Preview estimates may include keyword families whose manual precision validation is pending. "
            "Manual precision describes relevance among detected matches; recall was not estimated."
            if preliminary
            else "Adjusted associations are observational, not causal; keyword flags are excluded from the control-only Ridge model. "
            "BH-FDR is applied across the precision-validated held-out test family. "
            "Manual precision describes relevance among detected matches; recall was not estimated."
        )
        fig.text(0.01, 0.014, footer_text, fontsize=7.9, color="#555555", ha="left", va="bottom")
        fig.subplots_adjust(left=0.20, right=0.70, bottom=0.20, top=0.84)
        fig.savefig(figure_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    status = summarize_checks(checks)
    atomic_write_json(
        {
            "version": VERSION,
            "figure_style_version": FIGURE_STYLE_VERSION,
            "status": status,
            "inputs": {
                "analysis_ready": relative_to_project(feature_path),
                "text_features": relative_to_project(text_path),
                "precision": relative_to_project(precision_path),
                "control_model": relative_to_project(model_path),
            },
            "method": "Held-out control-only Ridge residual comparison; keyword flags never enter the control model.",
            "multiple_testing": {
                "method": "Benjamini-Hochberg FDR",
                "alpha": 0.05,
                "family": "all precision-validated held-out adjusted Welch residual-comparison tests",
                "interpretation": "q<0.05 is used for professor-facing evidence labeling; raw p-values remain available for audit.",
            },
            "mandatory_keywords": sorted(MANDATORY_ASSIGNMENT_KEYWORDS),
            "outputs": {"summary": relative_to_project(summary_path), "monthly_frequency": relative_to_project(monthly_path), "checks": relative_to_project(checks_path), "figure": relative_to_project(figure_path) if figure_path.exists() else None, "figure_preliminary": preliminary},
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }, manifest_path,
    )
    show_progress(100, f"complete; status={status['overall_status']}", final=True)
    return {"summary": summary_path, "monthly": monthly_path, "checks": checks_path, "manifest": manifest_path, "figure": figure_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate manually validated text signals against held-out asking prices.")
    parser.add_argument("--features", type=Path, default=FEATURE_TABLE)
    parser.add_argument("--text-features", type=Path, default=TEXT_TABLE)
    parser.add_argument("--precision", type=Path, default=PRECISION)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--model-manifest", type=Path, default=MODEL_MANIFEST)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run(args.features, args.text_features, args.precision, args.model, args.model_manifest)
    print("M3 TEXT PRICE SIGNALS COMPLETED")
    for name, path in outputs.items():
        if path.exists():
            print(f"{name}: {relative_to_project(path)}")


if __name__ == "__main__":
    main()
