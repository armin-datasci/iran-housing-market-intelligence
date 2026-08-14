from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd
import polars as pl

from src.common.config import configured_path, setting
from src.common.io_utils import atomic_write_csv, atomic_write_json
from src.common.paths import OUTPUTS_DIR, relative_to_project
from src.common.validation import Check, checks_frame, make_check, summarize_checks

VERSION = "m3-monthly-market-v1.1-apartment-temperature-aligned"
FIGURE_STYLE_VERSION = "monthly-market-figures-v2"
PROGRESS_WIDTH = 30
FIGURE_SIZE = (11.5, 6.2)
FIGURE_DPI = 300

SALES_BASE = OUTPUTS_DIR / "model_artifacts" / "milestone_3" / "analysis_populations" / "sales_analysis_base.parquet"
ACTIVE_REGIMES = [
    "sale", "rent_plus_deposit", "full_deposit", "rent_only", "rent_negotiable",
    "rent_unknown_or_incomplete", "temporary_rent",
]
RESIDENTIAL_FAMILIES = ["apartment", "villa"]
APARTMENT_FAMILY = "apartment"

# Presentation-only constants; analytical outputs do not depend on these values.
PRIMARY = "#2F6B9A"
RAW = "#9AA0A6"
BAND = "#B9CBDA"
IQR = "#BFD0DF"
TEXT = "#222222"
MUTED = "#5F6368"
GRID = "#DADCE0"


def show_progress(percent: int, label: str, *, final: bool = False) -> None:
    pct = max(0, min(100, int(percent)))
    filled = int(PROGRESS_WIDTH * pct / 100)
    bar = "#" * filled + "-" * (PROGRESS_WIDTH - filled)
    print(
        f"\rM3 monthly [{bar}] {pct:3d}% complete | {100-pct:3d}% remaining | {label[:42]:42s}",
        end="\n" if final else "", flush=True,
    )


def _b(name: str) -> pl.Expr:
    return pl.col(name).cast(pl.Boolean, strict=False).fill_null(False)


def _entity_frames(scan: pl.LazyFrame, aggs: list[pl.Expr], *, scope: str) -> pl.LazyFrame:
    national = scan.group_by("analysis_month").agg(aggs).with_columns([
        pl.lit("national").alias("entity_level"),
        pl.lit(None, dtype=pl.String).alias("city_slug"),
        pl.lit(None, dtype=pl.String).alias("neighborhood_slug"),
        pl.lit(scope).alias("market_scope"),
    ])
    city = scan.group_by(["analysis_month", "city_slug"]).agg(aggs).with_columns([
        pl.lit("city").alias("entity_level"),
        pl.lit(None, dtype=pl.String).alias("neighborhood_slug"),
        pl.lit(scope).alias("market_scope"),
    ])
    neighborhood = (
        scan.filter(pl.col("neighborhood_slug").is_not_null())
        .group_by(["analysis_month", "city_slug", "neighborhood_slug"])
        .agg(aggs)
        .with_columns([
            pl.lit("neighborhood").alias("entity_level"),
            pl.lit(scope).alias("market_scope"),
        ])
    )
    return pl.concat([national, city, neighborhood], how="diagonal_relaxed")


def _add_mom(frame: pl.DataFrame, value: str, output: str) -> pl.DataFrame:
    keys = ["market_scope", "entity_level", "city_slug", "neighborhood_slug"]
    return (
        frame.sort([*keys, "analysis_month"])
        .with_columns(pl.col(value).shift(1).over(keys).alias("_previous"))
        .with_columns(
            pl.when(pl.col("_previous").is_not_null() & (pl.col("_previous") != 0))
            .then((pl.col(value) / pl.col("_previous") - 1.0) * 100.0)
            .otherwise(None)
            .alias(output)
        )
        .drop("_previous")
    )


def _national_frame(frame: pl.DataFrame, scope: str) -> pd.DataFrame:
    out = (
        frame.filter((pl.col("market_scope") == scope) & (pl.col("entity_level") == "national"))
        .sort("analysis_month")
        .to_pandas()
    )
    if not out.empty:
        out["analysis_month"] = out["analysis_month"].astype(str).str[:7]
    return out


def _style_axis(ax: plt.Axes, months: list[str], ylabel: str, formatter: FuncFormatter) -> None:
    ax.set_xticks(range(len(months)), months)
    ax.set_xlabel("Month", color=TEXT)
    ax.set_ylabel(ylabel, color=TEXT)
    ax.yaxis.set_major_formatter(formatter)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, alpha=0.65, linewidth=0.8)
    ax.tick_params(colors=TEXT)
    ax.margins(x=0.02)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)


def _header(fig: plt.Figure, title: str, subtitle: str) -> None:
    fig.suptitle(title, x=0.075, y=0.975, ha="left", fontsize=15.5, fontweight="bold", color=TEXT)
    fig.text(0.075, 0.915, subtitle, ha="left", va="top", fontsize=9.4, color=MUTED)


def _save(fig: plt.Figure, path: Path, footer: str) -> None:
    fig.text(0.075, 0.025, footer, ha="left", va="bottom", fontsize=8.3, color=MUTED)
    fig.tight_layout(rect=(0.055, 0.075, 0.985, 0.865))
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_supply(national: pd.DataFrame, path: Path) -> None:
    if national.empty:
        return

    months = national["analysis_month"].tolist()
    x = list(range(len(months)))
    raw = national["raw_listing_count"].to_numpy(dtype=float)
    dedup = national["deduplicated_listing_count"].to_numpy(dtype=float)
    removed = raw - dedup
    raw_total, dedup_total, removed_total = raw.sum(), dedup.sum(), removed.sum()
    removed_rate = float(removed_total / raw_total * 100.0) if raw_total else 0.0

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    ax.plot(x, raw, marker="o", markersize=5, linewidth=1.6, linestyle="--", color=RAW,
            label="Raw listings", zorder=2)
    ax.plot(x, dedup, marker="o", markersize=5.5, linewidth=2.8, color=PRIMARY,
            label="Deduplicated listings", zorder=3)
    # Show the tiny adjustment without promoting it to a third legend series.
    ax.fill_between(x, dedup, raw, color=BAND, alpha=0.28, linewidth=0, zorder=1)

    _style_axis(ax, months, "Residential-sale listings", FuncFormatter(lambda y, _: f"{y / 1000:,.0f}k"))
    _header(
        fig,
        "Residential-Sale Listing Supply Proxy",
        f"Period totals: raw {int(raw_total):,} | deduplicated {int(dedup_total):,} | "
        f"removed {int(removed_total):,} ({removed_rate:.2f}%)",
    )

    low, high = float(min(raw.min(), dedup.min())), float(max(raw.max(), dedup.max()))
    pad = max((high - low) * 0.16, high * 0.004, 1.0)
    ax.set_ylim(max(0.0, low - pad), high + pad)

    for xi, value in zip(x, dedup):
        ax.annotate(
            f"{value / 1000:.1f}k", (xi, value), xytext=(0, 8), textcoords="offset points",
            ha="center", va="bottom", fontsize=8.3, color=PRIMARY,
            bbox={"boxstyle": "round,pad=0.16", "fc": "white", "ec": "none", "alpha": 0.82},
            zorder=4,
        )
    ax.legend(frameon=False, ncol=2, loc="upper left", bbox_to_anchor=(0, 1.02), borderaxespad=0)
    _save(
        fig, path,
        "Listing-flow proxy, not physical housing inventory, liquidity, or absorption. "
        "Deduplication uses canonical M2 supply-eligibility flags.",
    )


def _plot_price(national: pd.DataFrame, path: Path) -> None:
    if national.empty:
        return

    months = national["analysis_month"].tolist()
    x = list(range(len(months)))
    median = national["median_asking_price_per_sqm_toman"].to_numpy(dtype=float) / 1_000_000.0
    p25 = national["p25_asking_price_per_sqm_toman"].to_numpy(dtype=float) / 1_000_000.0
    p75 = national["p75_asking_price_per_sqm_toman"].to_numpy(dtype=float) / 1_000_000.0
    total_n = int(national["price_listing_n"].sum())

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    ax.fill_between(x, p25, p75, color=IQR, alpha=0.34, linewidth=0,
                    label="Interquartile range (P25-P75)", zorder=1)
    ax.plot(x, median, marker="o", markersize=5.8, linewidth=3.0, color=PRIMARY,
            label="Monthly median", zorder=3)

    _style_axis(ax, months, "Asking PSM (million toman / m2)", FuncFormatter(lambda y, _: f"{y:,.0f}"))
    _header(
        fig,
        "Apartment-Sale Asking PSM Trend",
        f"Canonical apartment-sale PSM eligible listings | total monthly observations: {total_n:,}",
    )

    low, high = float(p25.min()), float(p75.max())
    pad = max((high - low) * 0.08, 0.5)
    ax.set_ylim(max(0.0, low - pad), high + pad)

    for xi, value in zip(x, median):
        ax.annotate(
            f"{value:.1f}M", (xi, value), xytext=(0, 8), textcoords="offset points",
            ha="center", va="bottom", fontsize=8.3, color=PRIMARY,
            bbox={"boxstyle": "round,pad=0.16", "fc": "white", "ec": "none", "alpha": 0.86},
            zorder=4,
        )
    ax.legend(frameon=False, ncol=2, loc="upper left", bbox_to_anchor=(0, 1.02), borderaxespad=0)
    _save(
        fig, path,
        "Asking prices, not transactions. The shaded band is cross-sectional P25-P75 dispersion, "
        "not a confidence interval. Monetary scale is operationally treated as toman_assumed_unconfirmed.",
    )


def run(silver_path: Path | None = None, sales_base: Path = SALES_BASE, min_n: int | None = None) -> dict[str, Path]:
    silver = (silver_path or configured_path("silver_master")).resolve()
    sales_base = sales_base.resolve()
    for path in (silver, sales_base):
        if not path.exists():
            raise FileNotFoundError(f"Monthly-market input not found: {path}")

    min_n = int(min_n if min_n is not None else setting("analysis", "minimum_valid_listings", "default", default=30))
    if min_n <= 0:
        raise ValueError("minimum valid listings must be greater than zero")

    table_dir = OUTPUTS_DIR / "tables" / "milestone_3" / "monthly_market"
    qa_dir = OUTPUTS_DIR / "qa" / "milestone_3" / "monthly_market"
    fig_dir = OUTPUTS_DIR / "figures" / "milestone_3" / "monthly_market"
    for directory in (table_dir, qa_dir, fig_dir):
        directory.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    show_progress(0, "validating inputs")

    silver_scan = pl.scan_parquet(silver)
    required = {
        "analysis_month", "city_slug", "neighborhood_slug", "property_family", "price_regime",
        "is_core_analysis_period", "supply_analysis_eligible_flag",
    }
    missing = sorted(required - set(silver_scan.collect_schema().names()))
    if missing:
        raise ValueError(f"Silver Master missing monthly-market columns: {missing}")

    # Narrow scans reduce I/O while preserving exactly the same analytical populations.
    core = silver_scan.select(sorted(required)).filter(_b("is_core_analysis_period"))
    all_market = core.filter(pl.col("price_regime").is_in(ACTIVE_REGIMES))
    residential_sale = core.filter(
        (pl.col("price_regime") == "sale") & pl.col("property_family").is_in(RESIDENTIAL_FAMILIES)
    )
    apartment_sale = core.filter(
        (pl.col("price_regime") == "sale") & (pl.col("property_family") == APARTMENT_FAMILY)
    )

    supply_aggs = [
        pl.len().alias("raw_listing_count"),
        _b("supply_analysis_eligible_flag").sum().alias("deduplicated_listing_count"),
    ]
    supply = pl.concat([
        _entity_frames(all_market, supply_aggs, scope="all_market"),
        _entity_frames(residential_sale, supply_aggs, scope="residential_sale"),
        _entity_frames(apartment_sale, supply_aggs, scope="apartment_sale"),
    ], how="diagonal_relaxed").collect(engine="streaming")
    supply = _add_mom(supply, "deduplicated_listing_count", "deduplicated_supply_mom_pct")
    supply = _add_mom(supply, "raw_listing_count", "raw_supply_mom_pct")
    supply = supply.with_columns(
        (1.0 - pl.col("deduplicated_listing_count") / pl.col("raw_listing_count"))
        .fill_nan(0.0).fill_null(0.0).alias("duplicate_supply_reduction_rate")
    )
    show_progress(42, f"supply summaries: {supply.height:,} rows")

    sales = pl.scan_parquet(sales_base)
    sales_required = {
        "analysis_month", "city_slug", "neighborhood_slug", "property_family", "sale_price_per_sqm_final_toman",
    }
    missing_sales = sorted(sales_required - set(sales.collect_schema().names()))
    if missing_sales:
        raise ValueError(f"Sales base missing monthly-price columns: {missing_sales}")

    apartment_sales = sales.select(sorted(sales_required)).filter(pl.col("property_family") == APARTMENT_FAMILY)
    psm = pl.col("sale_price_per_sqm_final_toman").cast(pl.Float64, strict=False)
    price = _entity_frames(apartment_sales, [
        pl.len().alias("price_listing_n"),
        psm.median().alias("median_asking_price_per_sqm_toman"),
        psm.quantile(0.25).alias("p25_asking_price_per_sqm_toman"),
        psm.quantile(0.75).alias("p75_asking_price_per_sqm_toman"),
    ], scope="apartment_sale").collect(engine="streaming")
    price = _add_mom(price, "median_asking_price_per_sqm_toman", "median_price_mom_pct")
    price = price.with_columns((pl.col("price_listing_n") >= min_n).alias("price_reliable_flag"))
    show_progress(68, f"price summaries: {price.height:,} rows")

    supply_for_market = supply.filter(pl.col("market_scope") == "apartment_sale").drop("market_scope")
    price_for_market = price.filter(pl.col("market_scope") == "apartment_sale").drop("market_scope")
    keys = ["analysis_month", "entity_level", "city_slug", "neighborhood_slug"]
    market = (
        supply_for_market.join(price_for_market, on=keys, how="full", coalesce=True, nulls_equal=True)
        .with_columns([
            pl.lit("apartment_sale_proxy").alias("market_scope"),
            pl.lit("apartment_sale").alias("supply_population"),
            pl.lit("apartment_sale").alias("price_population"),
            pl.lit(VERSION).alias("analysis_version"),
        ])
        .sort(["entity_level", "city_slug", "neighborhood_slug", "analysis_month"])
    )

    # Presentation only: figures use the already-computed canonical summaries.
    supply_figure = fig_dir / "residential_sale_supply_raw_vs_deduplicated.png"
    price_figure = fig_dir / "apartment_sale_median_psm_monthly.png"
    _plot_supply(_national_frame(supply, "residential_sale"), supply_figure)
    _plot_price(_national_frame(price, "apartment_sale"), price_figure)

    apartment_supply_rows = supply.filter(pl.col("market_scope") == "apartment_sale").height
    population_mismatch_rows = market.filter(pl.col("supply_population") != pl.col("price_population")).height
    checks: list[Check] = [
        make_check("monthly_supply_nonempty", "monthly", supply.height, ">0", supply.height > 0),
        make_check("monthly_price_nonempty", "monthly", price.height, ">0", price.height > 0),
        make_check("monthly_market_nonempty", "monthly", market.height, ">0", market.height > 0),
        make_check("apartment_sale_supply_scope_nonempty", "monthly", apartment_supply_rows, ">0", apartment_supply_rows > 0),
        make_check(
            "temperature_population_alignment", "monthly", population_mismatch_rows, 0,
            population_mismatch_rows == 0,
            notes="Market-temperature input must compare apartment-sale supply with apartment-sale asking prices.",
        ),
    ]

    supply_path = table_dir / "monthly_supply_summary.csv"
    price_path = table_dir / "monthly_price_summary.csv"
    market_path = table_dir / "monthly_market_summary.csv"
    checks_path = qa_dir / "monthly_market_checks.csv"
    manifest_path = qa_dir / "monthly_market_manifest.json"

    atomic_write_csv(supply, supply_path)
    atomic_write_csv(price, price_path)
    atomic_write_csv(market, market_path)
    atomic_write_csv(checks_frame(checks), checks_path)
    status = summarize_checks(checks)
    atomic_write_json({
        "version": VERSION,
        "status": status,
        "minimum_valid_listings": min_n,
        "inputs": {"silver": relative_to_project(silver), "sales_base": relative_to_project(sales_base)},
        "outputs": {
            "supply": relative_to_project(supply_path),
            "price": relative_to_project(price_path),
            "market": relative_to_project(market_path),
            "checks": relative_to_project(checks_path),
            "supply_figure": relative_to_project(supply_figure) if supply_figure.exists() else None,
            "price_figure": relative_to_project(price_figure) if price_figure.exists() else None,
        },
        "notes": {
            "supply": "Listing-flow proxy, not inventory or absorption.",
            "price": "Apartment sale asking-price per sqm only.",
            "temperature_population": (
                "The joined monthly_market_summary uses apartment-sale supply and apartment-sale price on both sides. "
                "Residential-sale supply remains available only as a broader descriptive supply series."
            ),
            "figure_style_version": FIGURE_STYLE_VERSION,
            "figure_logic": "Presentation-only refactor; analytical definitions are unchanged.",
        },
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }, manifest_path)

    show_progress(100, f"complete in {time.perf_counter() - started:.1f}s", final=True)
    return {
        "supply": supply_path,
        "price": price_path,
        "market": market_path,
        "supply_figure": supply_figure,
        "price_figure": price_figure,
        "checks": checks_path,
        "manifest": manifest_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build canonical monthly supply and asking-price summaries.")
    parser.add_argument("--silver", type=Path, default=None)
    parser.add_argument("--sales-base", type=Path, default=SALES_BASE)
    parser.add_argument("--minimum-n", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run(args.silver, args.sales_base, args.minimum_n)
    print("M3 MONTHLY MARKET COMPLETED")
    for name, path in outputs.items():
        print(f"{name}: {relative_to_project(path)}")


if __name__ == "__main__":
    main()
