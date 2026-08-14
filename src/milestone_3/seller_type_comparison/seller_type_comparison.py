from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from scipy import stats
from sklearn.model_selection import StratifiedKFold

from src.common.config import setting
from src.common.io_utils import atomic_write_csv, atomic_write_json
from src.common.paths import OUTPUTS_DIR, relative_to_project
from src.common.validation import Check, checks_frame, make_check, summarize_checks
from src.milestone_3.price_drivers.price_drivers import active_features, build_pipeline, _prepare

VERSION = "m3-seller-comparison-v1"
FEATURE_TABLE = OUTPUTS_DIR / "model_artifacts" / "milestone_3" / "price_drivers" / "analysis_ready_features.parquet"
PRICE_DRIVER_MANIFEST = OUTPUTS_DIR / "qa" / "milestone_3" / "price_drivers" / "price_driver_manifest.json"
TARGET = "sale_price_per_sqm_final_toman"
LOG_TARGET = "log_sale_price_per_sqm"
AGENCY = "مشاور املاک"
PERSONAL = "شخصی"
PROGRESS_WIDTH = 30
FIGURE_STYLE_VERSION = "seller-figure-v2-primary-robustness"


def _style_comparison_axis(ax: plt.Axes) -> None:
    ax.grid(axis="x", alpha=0.16, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_color("#777777")
    ax.spines["bottom"].set_color("#777777")
    ax.tick_params(colors="#222222")


def _add_figure_header(fig: plt.Figure, title: str, subtitle: str) -> None:
    fig.text(0.08, 0.955, title, ha="left", va="top", fontsize=16.0, fontweight="bold", color="#111111")
    fig.text(0.08, 0.905, subtitle, ha="left", va="top", fontsize=9.5, color="#555555")


def _add_figure_footer(fig: plt.Figure, text: str) -> None:
    fig.text(0.015, 0.018, text, ha="left", va="bottom", fontsize=8.4, color="#555555")


def _save_figure(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout(rect=(0.075, 0.075, 0.985, 0.84))
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _pct_label(value: float) -> str:
    return f"{value:+.1f}%"


def show_progress(percent: int, label: str, *, final: bool = False) -> None:
    pct = max(0, min(100, int(percent)))
    filled = int(PROGRESS_WIDTH * pct / 100)
    bar = "#" * filled + "-" * (PROGRESS_WIDTH - filled)
    print(
        f"\rM3 seller [{bar}] {pct:3d}% complete | {100-pct:3d}% remaining | {label[:42]:42s}",
        end="\n" if final else "", flush=True,
    )


def _stratified_sample(frame: pd.DataFrame, maximum: int, seed: int) -> pd.DataFrame:
    if len(frame) <= maximum:
        return frame.copy()
    parts: list[pd.DataFrame] = []
    for _, group in frame.groupby("user_type", dropna=False):
        n = max(1, int(round(maximum * len(group) / len(frame))))
        parts.append(group.sample(n=min(n, len(group)), random_state=seed))
    out = pd.concat(parts, ignore_index=True)
    if len(out) > maximum:
        out = out.sample(n=maximum, random_state=seed)
    return out


def _diff_ci(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float, float]:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    diff = float(np.mean(a) - np.mean(b))
    se = math.sqrt(np.var(a, ddof=1) / len(a) + np.var(b, ddof=1) / len(b)) if len(a) > 1 and len(b) > 1 else np.nan
    lo = diff - 1.96 * se if np.isfinite(se) else np.nan
    hi = diff + 1.96 * se if np.isfinite(se) else np.nan
    p = float(stats.ttest_ind(a, b, equal_var=False, nan_policy="omit").pvalue) if len(a) > 1 and len(b) > 1 else np.nan
    return diff, lo, hi, p


def crossfit_residuals(frame: pd.DataFrame, features: list[str], alpha: float, min_frequency: int, folds: int, seed: int) -> np.ndarray:
    labels = frame["user_type"].astype(str).to_numpy()
    min_class = int(frame["user_type"].value_counts().min())
    n_splits = max(2, min(folds, min_class))
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    residuals = np.full(len(frame), np.nan, dtype=float)
    y = frame[LOG_TARGET].to_numpy(dtype=float)
    for train_idx, test_idx in splitter.split(frame, labels):
        pipe, _ = build_pipeline(features, alpha, min_frequency)
        pipe.fit(_prepare(frame.iloc[train_idx], features), y[train_idx])
        pred = pipe.predict(_prepare(frame.iloc[test_idx], features))
        residuals[test_idx] = y[test_idx] - pred
    return residuals


def stratified_comparison(frame: pd.DataFrame, minimum_seller_n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = frame.copy()
    work["area_bin"] = (pd.to_numeric(work.get("primary_area_sqm"), errors="coerce") / 20.0).round() * 20.0
    work["age_bin"] = (pd.to_numeric(work.get("building_age_years"), errors="coerce") / 10.0).round() * 10.0
    work["rooms_bin"] = pd.to_numeric(work.get("rooms_count_num"), errors="coerce").round()
    keys = [column for column in ["city_slug", "cat3_slug", "area_bin", "age_bin", "rooms_bin"] if column in work]
    grouped = work.groupby(keys + ["user_type"], dropna=False)[LOG_TARGET].agg(["count", "mean"]).reset_index()
    count_pivot = grouped.pivot_table(index=keys, columns="user_type", values="count", aggfunc="first", fill_value=0)
    mean_pivot = grouped.pivot_table(index=keys, columns="user_type", values="mean", aggfunc="first")
    if AGENCY not in count_pivot.columns or PERSONAL not in count_pivot.columns:
        return pd.DataFrame(), pd.DataFrame()
    valid = (count_pivot[AGENCY] >= minimum_seller_n) & (count_pivot[PERSONAL] >= minimum_seller_n)
    if valid.sum() == 0:
        return pd.DataFrame(), pd.DataFrame()
    detail = count_pivot.loc[valid].reset_index()
    means = mean_pivot.loc[valid]
    detail["agency_n"] = count_pivot.loc[valid, AGENCY].to_numpy(dtype=int)
    detail["personal_n"] = count_pivot.loc[valid, PERSONAL].to_numpy(dtype=int)
    detail["log_difference_agency_minus_personal"] = means[AGENCY].to_numpy(dtype=float) - means[PERSONAL].to_numpy(dtype=float)
    detail["stratum_weight"] = np.minimum(detail["agency_n"], detail["personal_n"])
    weighted_log = float(np.average(detail["log_difference_agency_minus_personal"], weights=detail["stratum_weight"]))
    pct = float((math.exp(weighted_log) - 1.0) * 100.0)
    summary = pd.DataFrame([{
        "method": "coarsened_similar_unit_strata",
        "price_difference_pct_agency_minus_personal": pct,
        "log_difference": weighted_log,
        "matched_strata_n": len(detail),
        "agency_rows_in_matched_strata": int(detail["agency_n"].sum()),
        "personal_rows_in_matched_strata": int(detail["personal_n"].sum()),
    }])
    return summary, detail


def run(feature_path: Path = FEATURE_TABLE) -> dict[str, Path]:
    feature_path = feature_path.resolve()
    if not feature_path.exists():
        raise FileNotFoundError(f"Analysis-ready features not found: {feature_path}")
    seed = int(setting("project", "random_seed", default=42))
    max_rows = int(setting("milestone_3", "seller_comparison", "max_crossfit_rows", default=150000))
    folds = int(setting("milestone_3", "seller_comparison", "crossfit_folds", default=5))
    minimum_seller_n = int(setting("milestone_3", "seller_comparison", "minimum_seller_n_per_stratum", default=5))
    min_frequency = int(setting("milestone_3", "models", "one_hot_min_frequency", default=50))
    alpha = float(setting("milestone_3", "seller_comparison", "ridge_alpha", default=10.0))
    if PRICE_DRIVER_MANIFEST.exists():
        try:
            alpha = float(json.loads(PRICE_DRIVER_MANIFEST.read_text(encoding="utf-8")).get("selected_alpha", alpha))
        except (ValueError, TypeError, json.JSONDecodeError):
            pass

    table_dir = OUTPUTS_DIR / "tables" / "milestone_3" / "seller_type_comparison"
    qa_dir = OUTPUTS_DIR / "qa" / "milestone_3" / "seller_type_comparison"
    fig_dir = OUTPUTS_DIR / "figures" / "milestone_3" / "seller_type_comparison"
    for d in [table_dir, qa_dir, fig_dir]:
        d.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    show_progress(0, "loading seller comparison population")

    frame = pd.read_parquet(feature_path)
    required = {"source_row_id", "user_type", TARGET, LOG_TARGET, "analysis_month", "city_slug", "cat3_slug"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Analysis-ready features missing seller columns: {missing}")
    population = frame.loc[frame["user_type"].isin([AGENCY, PERSONAL])].copy()
    counts = population["user_type"].value_counts()
    agency_n, personal_n = int(counts.get(AGENCY, 0)), int(counts.get(PERSONAL, 0))
    if agency_n < 20 or personal_n < 20:
        raise RuntimeError(f"Seller comparison requires both labels; agency={agency_n}, personal={personal_n}")
    show_progress(18, f"seller rows: agency={agency_n:,}, personal={personal_n:,}")

    a_price = population.loc[population["user_type"] == AGENCY, TARGET].to_numpy(dtype=float)
    p_price = population.loc[population["user_type"] == PERSONAL, TARGET].to_numpy(dtype=float)
    raw_pct = float((np.median(a_price) / np.median(p_price) - 1.0) * 100.0)
    raw_mw = stats.mannwhitneyu(a_price, p_price, alternative="two-sided", method="asymptotic")

    model_pop = _stratified_sample(population, max_rows, seed)
    numeric, binary, categorical = active_features(model_pop.columns)
    features = numeric + binary + categorical
    residuals = crossfit_residuals(model_pop, features, alpha, min_frequency, folds, seed)
    model_pop["log_residual"] = residuals
    a_res = model_pop.loc[model_pop["user_type"] == AGENCY, "log_residual"].to_numpy(dtype=float)
    p_res = model_pop.loc[model_pop["user_type"] == PERSONAL, "log_residual"].to_numpy(dtype=float)
    diff, lo, hi, adjusted_p = _diff_ci(a_res, p_res)
    adjusted_pct = float((math.exp(diff) - 1.0) * 100.0)
    adjusted_lo = float((math.exp(lo) - 1.0) * 100.0) if np.isfinite(lo) else np.nan
    adjusted_hi = float((math.exp(hi) - 1.0) * 100.0) if np.isfinite(hi) else np.nan
    show_progress(62, f"control-only cross-fit complete on {len(model_pop):,} rows")

    strat_summary, strat_detail = stratified_comparison(population, minimum_seller_n)
    strat_pct = float(strat_summary.loc[0, "price_difference_pct_agency_minus_personal"]) if not strat_summary.empty else np.nan
    show_progress(78, f"similar-unit strata: {len(strat_detail):,}")

    summary = pd.DataFrame([
        {
            "comparison": "agency_vs_personal",
            "agency_label": AGENCY,
            "personal_label": PERSONAL,
            "agency_n": agency_n,
            "personal_n": personal_n,
            "raw_median_difference_pct": raw_pct,
            "raw_mann_whitney_p_value": float(raw_mw.pvalue),
            "adjusted_crossfit_difference_pct": adjusted_pct,
            "adjusted_ci_low_pct": adjusted_lo,
            "adjusted_ci_high_pct": adjusted_hi,
            "adjusted_welch_p_value": adjusted_p,
            "stratified_difference_pct": strat_pct,
            "crossfit_rows": len(model_pop),
            "interpretation": "observational association after observed controls; not a causal agency premium/discount",
            "analysis_version": VERSION,
        }
    ])

    checks: list[Check] = [
        make_check("agency_population_nonempty", "seller", agency_n, ">=20", agency_n >= 20),
        make_check("personal_population_nonempty", "seller", personal_n, ">=20", personal_n >= 20),
        make_check("crossfit_residuals_complete", "seller", int(np.isfinite(residuals).sum()), len(model_pop), int(np.isfinite(residuals).sum()) == len(model_pop)),
        make_check(
            "causal_claim_guard", "method", "observational association", "non-causal", True,
            notes="Seller type is not randomly assigned; residual confounding remains possible.",
        ),
        make_check(
            "stratified_robustness_available", "seller", len(strat_detail), ">0", len(strat_detail) > 0,
            critical=False, review_on_fail=True,
        ),
    ]

    summary_path = table_dir / "seller_type_comparison_summary.csv"
    strat_path = qa_dir / "seller_type_stratified_detail.csv"
    checks_path = qa_dir / "seller_type_checks.csv"
    manifest_path = qa_dir / "seller_type_manifest.json"
    figure_path = fig_dir / "seller_type_raw_adjusted_stratified.png"
    atomic_write_csv(pl.from_pandas(summary), summary_path)
    atomic_write_csv(pl.from_pandas(strat_detail) if not strat_detail.empty else pl.DataFrame({"status": ["no_valid_strata"]}), strat_path)
    atomic_write_csv(checks_frame(checks), checks_path)

    labels = [
        "Raw median (unadjusted)",
        "Adjusted cross-fit (primary)",
        "Similar-unit strata (robustness)",
    ]
    effects = np.asarray([raw_pct, adjusted_pct, strat_pct], dtype=float)
    y = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(11.8, 6.2))
    point_colors = ["#9AA0A6", "#2F6B9A", "#4F7F63"]
    point_sizes = [66, 92, 70]
    for yi, value, color, size in zip(y, effects, point_colors, point_sizes):
        if np.isfinite(value):
            ax.scatter([value], [yi], s=size, color=color, zorder=4)

    ci_available = bool(
        np.isfinite(adjusted_lo)
        and np.isfinite(adjusted_hi)
        and adjusted_lo <= adjusted_pct <= adjusted_hi
    )
    if ci_available:
        ax.errorbar(
            [adjusted_pct],
            [1],
            xerr=[[adjusted_pct - adjusted_lo], [adjusted_hi - adjusted_pct]],
            fmt="none",
            ecolor="#2F6B9A",
            elinewidth=1.9,
            capsize=4.5,
            capthick=1.5,
            zorder=3,
        )

    ax.axvline(0, linewidth=1.25, color="#666666", zorder=1)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Agency minus personal asking-PSM difference (%)")
    _style_comparison_axis(ax)

    finite = effects[np.isfinite(effects)]
    x_candidates = finite.tolist()
    if ci_available:
        x_candidates.extend([adjusted_lo, adjusted_hi])
    data_min = min(x_candidates) if x_candidates else 0.0
    data_max = max(x_candidates) if x_candidates else 1.0
    data_span = max(data_max - min(data_min, 0.0), 1.0)
    x_left = min(-0.045 * data_span, data_min - 0.05 * data_span)
    x_right = data_max + 0.12 * data_span
    ax.set_xlim(x_left, x_right)

    label_offset = 0.016 * data_span
    for yi, value in enumerate(effects):
        if not np.isfinite(value):
            continue
        if yi == 1 and ci_available:
            label_x = max(value, adjusted_hi) + label_offset
            label_text = f"{_pct_label(value)}  [95% CI {adjusted_lo:+.1f}, {adjusted_hi:+.1f}]"
        else:
            label_x = value + label_offset
            label_text = _pct_label(value)
        ax.text(
            label_x,
            yi,
            label_text,
            va="center",
            ha="left",
            fontsize=9.0,
            color="#202020",
        )

    matched_strata_n = int(strat_summary.loc[0, "matched_strata_n"]) if not strat_summary.empty else 0
    subtitle = (
        f"Agency N={agency_n:,} | Personal N={personal_n:,} | "
        f"primary = cross-fitted adjusted contrast | robustness = {matched_strata_n:,} matched similar-unit strata"
    )
    _add_figure_header(fig, "Seller-Type Asking-PSM Comparison", subtitle)
    _add_figure_footer(
        fig,
        "Observational agency-vs-personal asking-PSM contrasts. The raw gap reflects composition; "
        "adjusted and stratified estimates reduce observed confounding but do not establish causality.",
    )
    _save_figure(fig, figure_path)

    status = summarize_checks(checks)
    atomic_write_json(
        {
            "version": VERSION,
            "status": status,
            "input": relative_to_project(feature_path),
            "labels": {"agency": AGENCY, "personal": PERSONAL},
            "method": "raw + control-only Ridge cross-fitting + coarsened similar-unit strata",
            "figure_style_version": FIGURE_STYLE_VERSION,
            "outputs": {"summary": relative_to_project(summary_path), "strata": relative_to_project(strat_path), "figure": relative_to_project(figure_path), "checks": relative_to_project(checks_path)},
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }, manifest_path,
    )
    show_progress(100, f"complete in {time.perf_counter()-started:.1f}s", final=True)
    return {"summary": summary_path, "strata": strat_path, "figure": figure_path, "checks": checks_path, "manifest": manifest_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare agency and personal asking prices for similar observed units.")
    parser.add_argument("--features", type=Path, default=FEATURE_TABLE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run(args.features)
    print("M3 SELLER TYPE COMPARISON COMPLETED")
    for name, path in outputs.items():
        print(f"{name}: {relative_to_project(path)}")


if __name__ == "__main__":
    main()
