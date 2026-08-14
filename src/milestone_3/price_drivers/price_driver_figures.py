from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from src.common.paths import OUTPUTS_DIR, relative_to_project

SUMMARY = OUTPUTS_DIR / "tables" / "milestone_3" / "price_drivers" / "price_driver_summary.csv"
PERMUTATION = OUTPUTS_DIR / "tables" / "milestone_3" / "price_drivers" / "price_driver_permutation_importance.csv"
DIAGNOSTICS = OUTPUTS_DIR / "tables" / "milestone_3" / "price_drivers" / "price_driver_model_diagnostics.csv"

FIGURE_STYLE_VERSION = "price-driver-figures-v1.1-final-analysis-presentation"
BLUE = "#2F6B9A"
GRAY = "#8A8F98"
DARK_GRAY = "#555555"
LIGHT_GRAY = "#7B8FA1"
CI_DISPLAY_TOLERANCE = 1e-4
ASSOCIATION_NEAR_ZERO = 0.005
PERMUTATION_NEAR_ZERO = 0.0005

FEATURE_LABELS = {
    "location": "Location controls (City + Neighborhood; grouped)",
    "property_type": "Property-type controls",
    "time": "Month / time control",
    "primary_area_sqm": "Area",
    "rooms_count_num": "Rooms",
    "building_age_years": "Building age",
    "floor_num": "Floor",
    "has_elevator_bool": "Elevator",
    "has_parking_bool": "Parking",
    "has_warehouse_bool": "Storage",
    "has_balcony_bool": "Balcony",
    "is_rebuilt_bool": "Rebuilt",
    "construction_year_before_1370_flag": "Built before 1370",
    "building_direction": "Building direction",
}

CONTROL_BLOCKS = {"location", "property_type", "time"}


def _feature_label(value: object) -> str:
    text = str(value)
    if ":" in text:
        base, level = text.split(":", 1)
        base_label = FEATURE_LABELS.get(base, base.replace("_", " ").title())
        level_label = (
            "Unspecified"
            if level.lower() in {"unselect", "unknown", "none", "nan"}
            else level.replace("-", " ").title()
        )
        return f"{base_label}: {level_label}"
    return FEATURE_LABELS.get(text, text.replace("_", " ").title())


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _numeric(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)


def _test_n(diagnostics: pd.DataFrame) -> int:
    if not {"split", "n"}.issubset(diagnostics.columns):
        return 0
    test_rows = diagnostics.loc[diagnostics["split"].astype(str).str.lower().eq("test"), "n"]
    if test_rows.empty:
        return 0
    value = pd.to_numeric(test_rows, errors="coerce").dropna()
    return int(value.iloc[0]) if not value.empty else 0


def _style_axis(ax: plt.Axes, *, grid_axis: str = "x") -> None:
    ax.set_axisbelow(True)
    ax.grid(axis=grid_axis, alpha=0.18, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=9.5)


def _finalize_figure(
    fig: plt.Figure,
    ax: plt.Axes,
    *,
    title: str,
    subtitle: str,
    footer: str,
    output_path: Path,
) -> Path:
    # Reserve explicit header/footer zones so title, subtitle, and notes never collide
    # with the plotting area, regardless of the number or length of category labels.
    fig.tight_layout(rect=(0, 0.105, 1, 0.855))
    left = ax.get_position().x0
    fig.text(left, 0.965, title, ha="left", va="top", fontsize=15.5, fontweight="bold")
    fig.text(left, 0.925, subtitle, ha="left", va="top", fontsize=9.5, color=DARK_GRAY)
    fig.text(0.01, 0.025, footer, ha="left", va="bottom", fontsize=8.4, color=DARK_GRAY)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def _association_label(value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    if abs(value) < ASSOCIATION_NEAR_ZERO:
        return "≈0.00%"
    if abs(value) < 0.1:
        return f"{value:+.2f}%"
    return f"{value:+.1f}%"


def _permutation_label(value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    if abs(value) < PERMUTATION_NEAR_ZERO:
        return "≈0.000"
    return f"{value:.3f}"


def _plot_adjusted_associations(summary: pd.DataFrame, test_n: int, out_dir: Path) -> Path:
    work = summary.copy()
    work["adjusted_effect_pct"] = pd.to_numeric(work["adjusted_effect_pct"], errors="coerce")
    work = work[np.isfinite(work["adjusted_effect_pct"])].copy()
    work = work.sort_values("adjusted_effect_pct").reset_index(drop=True)
    if work.empty:
        raise ValueError("No finite adjusted-effect estimates were available for the association figure.")

    assoc_path = out_dir / "adjusted_asking_price_associations.png"
    fig, ax = plt.subplots(figsize=(11.5, max(6.6, 0.46 * len(work))))
    y = np.arange(len(work))
    effects = work["adjusted_effect_pct"].to_numpy(dtype=float)
    labels = [_feature_label(value) for value in work["feature"].astype(str)]

    ci_low = effects.copy()
    ci_high = effects.copy()
    informative_ci = np.zeros(len(work), dtype=bool)
    ci_columns = {"adjusted_effect_ci_low", "adjusted_effect_ci_high"}
    if ci_columns.issubset(work.columns):
        ci_low = _numeric(work["adjusted_effect_ci_low"])
        ci_high = _numeric(work["adjusted_effect_ci_high"])
        informative_ci = (
            np.isfinite(ci_low)
            & np.isfinite(ci_high)
            & (ci_high >= ci_low)
            & ((ci_high - ci_low) > CI_DISPLAY_TOLERANCE)
        )

    ax.scatter(effects, y, s=46, color=BLUE, zorder=3)
    if informative_ci.any():
        idx = np.flatnonzero(informative_ci)
        xerr = np.vstack(
            [
                np.maximum(effects[idx] - ci_low[idx], 0.0),
                np.maximum(ci_high[idx] - effects[idx], 0.0),
            ]
        )
        ax.errorbar(
            effects[idx],
            y[idx],
            xerr=xerr,
            fmt="none",
            ecolor=LIGHT_GRAY,
            elinewidth=1.35,
            capsize=3,
            zorder=2,
        )

    ax.set_yticks(y, labels)
    ax.axvline(0, linewidth=1.2, color="#666666")
    ax.set_xlabel("Model-implied adjusted difference in asking PSM (%)")
    _style_axis(ax, grid_axis="x")

    finite_bounds = np.concatenate(
        [
            effects[np.isfinite(effects)],
            ci_low[informative_ci],
            ci_high[informative_ci],
        ]
    )
    data_min = float(np.nanmin(finite_bounds))
    data_max = float(np.nanmax(finite_bounds))
    span = max(data_max - data_min, 1.0)
    ax.set_xlim(data_min - 0.10 * span, data_max + 0.16 * span)
    label_pad = 0.014 * span

    for yi, value in enumerate(effects):
        ax.text(
            value + (label_pad if value >= 0 else -label_pad),
            yi,
            _association_label(value),
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=8.5,
        )

    subtitle = (
        f"Held-out reference N={test_n:,} | numeric: P75 vs P25 | binary: true vs false | "
        "categorical: level vs model reference"
    )
    if informative_ci.any():
        footer = (
            "Observational, model-implied contrasts; non-degenerate bootstrap intervals are shown where available. "
            "Not causal effects and not shares of price. 'Unspecified' categories may partly capture missingness/selection."
        )
    else:
        footer = (
            "Observational, model-implied contrasts; available intervals are degenerate or not informative and are not plotted. "
            "Not causal effects and not shares of price. 'Unspecified' categories may partly capture missingness/selection."
        )

    return _finalize_figure(
        fig,
        ax,
        title="Adjusted Asking-Price Association by Property Characteristic",
        subtitle=subtitle,
        footer=footer,
        output_path=assoc_path,
    )


def _plot_predictive_contribution(permutation: pd.DataFrame, test_n: int, out_dir: Path) -> Path:
    work = permutation.copy()
    work["heldout_rmse_log_increase_mean"] = pd.to_numeric(
        work["heldout_rmse_log_increase_mean"], errors="coerce"
    )
    work = work[np.isfinite(work["heldout_rmse_log_increase_mean"])].copy()
    work = work.sort_values("heldout_rmse_log_increase_mean").reset_index(drop=True)
    if work.empty:
        raise ValueError("No finite held-out permutation estimates were available.")

    perm_path = out_dir / "heldout_predictive_contribution.png"
    fig, ax = plt.subplots(figsize=(11.5, max(6.2, 0.43 * len(work))))
    y = np.arange(len(work))
    means = work["heldout_rmse_log_increase_mean"].to_numpy(dtype=float)
    blocks = work["feature_block"].astype(str).tolist()
    labels = [_feature_label(value) for value in blocks]
    colors = [GRAY if block in CONTROL_BLOCKS else BLUE for block in blocks]
    bars = ax.barh(y, means, color=colors, zorder=2)

    if "heldout_rmse_log_increase_sd" in work.columns:
        sd = pd.to_numeric(work["heldout_rmse_log_increase_sd"], errors="coerce").fillna(0).to_numpy(dtype=float)
        sd = np.maximum(sd, 0.0)
        if np.any(sd > 0):
            ax.errorbar(
                means,
                y,
                xerr=sd,
                fmt="none",
                ecolor=DARK_GRAY,
                elinewidth=1.05,
                capsize=2.5,
                zorder=3,
            )

    ax.set_yticks(y, labels)
    ax.set_xlabel("Increase in held-out RMSE (log asking PSM) after permutation")
    ax.axvline(0, linewidth=1.0, color="#666666")
    _style_axis(ax, grid_axis="x")

    data_min = float(np.nanmin(means))
    data_max = float(np.nanmax(means))
    span = max(data_max - data_min, max(abs(data_max), 1e-6))
    left_pad = 0.05 * span if data_min < 0 else 0.012 * span
    right_pad = 0.08 * span
    ax.set_xlim(min(0.0, data_min - left_pad), data_max + right_pad)
    label_pad = 0.010 * span

    for bar, value in zip(bars, means):
        ax.text(
            value + (label_pad if value >= 0 else -label_pad),
            bar.get_y() + bar.get_height() / 2,
            _permutation_label(value),
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=8.5,
        )

    ax.legend(
        handles=[
            Patch(facecolor=BLUE, label="Property characteristic"),
            Patch(facecolor=GRAY, label="Model control block"),
        ],
        loc="lower right",
        frameon=False,
        fontsize=8.5,
    )

    subtitle = (
        f"Test sample up to N={test_n:,} | City + Neighborhood permuted together as one location block | "
        "linear scale preserved"
    )
    footer = (
        "Grouped location permutation avoids impossible City/Neighborhood combinations. Gray bars are model controls. "
        "Values near zero indicate no measurable held-out contribution at this precision; predictive contribution is not causal importance."
    )

    return _finalize_figure(
        fig,
        ax,
        title="Held-out Predictive Contribution",
        subtitle=subtitle,
        footer=footer,
        output_path=perm_path,
    )


def run(
    summary_path: Path = SUMMARY,
    permutation_path: Path = PERMUTATION,
    diagnostics_path: Path = DIAGNOSTICS,
) -> dict[str, Path]:
    for path in [summary_path, permutation_path, diagnostics_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required price-driver output not found: {path}")

    summary = pd.read_csv(summary_path)
    permutation = pd.read_csv(permutation_path)
    diagnostics = pd.read_csv(diagnostics_path)

    _require_columns(summary, {"feature", "adjusted_effect_pct"}, summary_path.name)
    _require_columns(
        permutation,
        {"feature_block", "heldout_rmse_log_increase_mean"},
        permutation_path.name,
    )

    test_n = _test_n(diagnostics)
    out_dir = OUTPUTS_DIR / "figures" / "milestone_3" / "price_drivers"
    out_dir.mkdir(parents=True, exist_ok=True)

    assoc_path = _plot_adjusted_associations(summary, test_n, out_dir)
    perm_path = _plot_predictive_contribution(permutation, test_n, out_dir)

    return {
        "adjusted_associations": assoc_path,
        "predictive_contribution": perm_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build two complementary technical price-driver figures.")
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--permutation", type=Path, default=PERMUTATION)
    parser.add_argument("--diagnostics", type=Path, default=DIAGNOSTICS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run(args.summary, args.permutation, args.diagnostics)
    print(f"M3 PRICE DRIVER FIGURES COMPLETED | style={FIGURE_STYLE_VERSION}")
    for name, path in outputs.items():
        print(f"{name}: {relative_to_project(path)}")


if __name__ == "__main__":
    main()
