from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

from src.common.config import setting
from src.common.io_utils import atomic_write_csv, atomic_write_json
from src.common.paths import OUTPUTS_DIR, relative_to_project
from src.common.validation import Check, checks_frame, make_check, summarize_checks

VERSION = "m3-market-temperature-v1.3-frozen-final"
MONTHLY_MARKET = OUTPUTS_DIR / "tables" / "milestone_3" / "monthly_market" / "monthly_market_summary.csv"
MARKET_MAP = OUTPUTS_DIR / "tables" / "milestone_3" / "market_map" / "neighborhood_market_summary.csv"
PROGRESS_WIDTH = 30
FOUR_CITY_SLUGS = ("tehran", "mashhad", "karaj", "isfahan")
LEGACY_MIN_MONTHS = 4
LEGACY_MIN_MOM = 3


def _display_slug(value: object) -> str:
    text = str(value or "").replace("-", " ").replace("_", " ").strip()
    return " ".join(part.capitalize() for part in text.split())


def show_progress(percent: int, label: str, *, final: bool = False) -> None:
    pct = max(0, min(100, int(percent)))
    filled = int(PROGRESS_WIDTH * pct / 100)
    bar = "#" * filled + "-" * (PROGRESS_WIDTH - filled)
    print(
        f"\rM3 temperature [{bar}] {pct:3d}% complete | {100-pct:3d}% remaining | {label[:42]:42s}",
        end="\n" if final else "",
        flush=True,
    )


def _trend(values: pd.Series, months: pd.Series) -> tuple[float | None, float | None]:
    numeric = pd.to_numeric(values, errors="coerce")
    parsed = pd.to_datetime(months.astype(str).str[:7] + "-01", errors="coerce")
    mask = numeric.notna() & numeric.gt(0) & parsed.notna()
    if int(mask.sum()) < 2:
        return None, None
    clean = numeric.loc[mask].to_numpy(dtype=float)
    month_values = parsed.loc[mask]
    x = month_values.dt.year.to_numpy(dtype=float) * 12.0 + month_values.dt.month.to_numpy(dtype=float)
    x = x - x.min()
    y = np.log(clean)
    slope = float(np.polyfit(x, y, 1)[0])
    monthly_pct = float((math.exp(slope) - 1.0) * 100.0)
    ordered = np.argsort(x)
    deltas = np.diff(y[ordered])
    if len(deltas) == 0 or abs(slope) < 1e-12:
        consistency = 0.5
    else:
        consistency = float(np.mean(np.sign(deltas) == np.sign(slope)))
    return monthly_pct, consistency


def _bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.astype("string").str.strip().str.lower().isin(["true", "1", "yes", "y"])


def _rank01(series: pd.Series) -> pd.Series:
    if series.notna().sum() <= 1:
        return pd.Series(0.5, index=series.index, dtype=float)
    return series.rank(method="average", pct=True).fillna(0.5)


def _rank_scaled_minus1_plus1(series: pd.Series) -> pd.Series:
    n = int(series.notna().sum())
    if n <= 1:
        return pd.Series(0.0, index=series.index, dtype=float)
    return 2.0 * ((series.rank(method="average") - 1.0) / float(n - 1)) - 1.0


def _rank_scaled_zero_one(series: pd.Series) -> pd.Series:
    n = int(series.notna().sum())
    if n <= 1:
        return pd.Series(0.5, index=series.index, dtype=float)
    return (series.rank(method="average") - 1.0) / float(n - 1)


def _score(
    features: pd.DataFrame,
    price_weight: float,
    supply_weight: float,
    hot_q: float,
    cold_q: float,
    label: str,
    *,
    supply_signal: str = "activity",
    gate_mode: str = "price_only",
) -> pd.DataFrame:
    out = features.copy()
    out["price_rank"] = out.groupby("entity_level", dropna=False)["price_trend_pct_per_month"].transform(_rank01)
    if supply_signal == "activity":
        supply_column = "supply_trend_pct_per_month"
    elif supply_signal == "tightness":
        supply_column = "supply_tightness_pct_per_month"
    else:
        raise ValueError(f"Unsupported supply_signal: {supply_signal}")
    out["supply_signal_rank"] = out.groupby("entity_level", dropna=False)[supply_column].transform(_rank01)
    weighted = price_weight * out["price_rank"] + supply_weight * out["supply_signal_rank"]
    score_col = f"temperature_score_{label}"
    pct_col = f"temperature_percentile_{label}"
    label_col = f"temperature_label_{label}"
    rank_col = f"temperature_rank_{label}"
    out[score_col] = 200.0 * (weighted - 0.5) * out["evidence_multiplier"]
    out[pct_col] = out.groupby("entity_level", dropna=False)[score_col].rank(method="average", pct=True)
    pct = out[pct_col]
    if gate_mode == "price_only":
        hot_gate = out["price_trend_pct_per_month"] > 0
        cold_gate = out["price_trend_pct_per_month"] < 0
    elif gate_mode == "price_and_supply":
        hot_gate = (out["price_trend_pct_per_month"] > 0) & (out["supply_trend_pct_per_month"] <= 0)
        cold_gate = (out["price_trend_pct_per_month"] < 0) & (out["supply_trend_pct_per_month"] >= 0)
    elif gate_mode == "none":
        hot_gate = pd.Series(True, index=out.index)
        cold_gate = pd.Series(True, index=out.index)
    else:
        raise ValueError(f"Unsupported gate_mode: {gate_mode}")
    hot = (pct >= hot_q) & hot_gate
    cold = (pct <= cold_q) & cold_gate
    out[label_col] = np.select([hot, cold], ["HOT", "COLD"], default="NEUTRAL")
    out[rank_col] = out.groupby("entity_level", dropna=False)[score_col].rank(method="first", ascending=False)
    return out


def build_features(monthly: pd.DataFrame, min_months: int) -> pd.DataFrame:
    keys = ["entity_level", "city_slug", "neighborhood_slug"]
    rows: list[dict[str, Any]] = []
    for key, group in monthly.groupby(keys, dropna=False, sort=False):
        group = group.sort_values("analysis_month")
        price_group = group.loc[_bool_series(group["price_reliable_flag"])] if "price_reliable_flag" in group else group
        price_trend, price_consistency = _trend(price_group["median_asking_price_per_sqm_toman"], price_group["analysis_month"])
        supply_trend, supply_consistency = _trend(group["deduplicated_listing_count"], group["analysis_month"])
        if price_trend is None or supply_trend is None:
            continue
        price_months = int(pd.to_numeric(price_group["median_asking_price_per_sqm_toman"], errors="coerce").notna().sum())
        supply_months = int(pd.to_numeric(group["deduplicated_listing_count"], errors="coerce").notna().sum())
        usable_months = min(price_months, supply_months)
        if usable_months < min_months:
            continue
        coverage_score = min(1.0, usable_months / float(max(4, min_months)))
        consistency_values = [value for value in [price_consistency, supply_consistency] if value is not None and np.isfinite(value)]
        trend_stability = float(np.mean(consistency_values)) if consistency_values else 0.5
        evidence = 0.65 + 0.20 * coverage_score + 0.15 * trend_stability
        if "price_listing_n" in price_group.columns:
            sample_size = int(pd.to_numeric(price_group["price_listing_n"], errors="coerce").fillna(0).sum())
        else:
            sample_size = int(pd.to_numeric(group["deduplicated_listing_count"], errors="coerce").fillna(0).sum())
        median_monthly_n = float(pd.to_numeric(group["deduplicated_listing_count"], errors="coerce").median())
        rows.append(
            {
                "entity_level": key[0],
                "city_slug": key[1],
                "neighborhood_slug": key[2],
                "price_trend_pct_per_month": price_trend,
                "supply_trend_pct_per_month": supply_trend,
                "supply_tightness_pct_per_month": -supply_trend,
                "price_direction_consistency": price_consistency,
                "supply_direction_consistency": supply_consistency,
                "trend_stability_score": trend_stability,
                "price_month_count": price_months,
                "supply_month_count": supply_months,
                "usable_month_count": usable_months,
                "sample_size": sample_size,
                "median_monthly_listing_count": median_monthly_n,
                "evidence_multiplier": float(min(1.0, max(0.0, evidence))),
            }
        )
    return pd.DataFrame(rows)


def _professor_facing_neighborhoods(
    base: pd.DataFrame,
    *,
    minimum_months: int = 5,
    minimum_sample_size: int = 100,
) -> pd.DataFrame:
    """Return reliable four-city neighborhood rows for surfaced HOT/COLD extrema only.

    This gate does not change the canonical all-city ranking or labels. It only prevents
    short/low-N histories from being promoted into the professor-facing top-5 figure.
    """
    if base.empty:
        return base.copy()
    required = {
        "entity_level",
        "city_slug",
        "neighborhood_slug",
        "price_month_count",
        "supply_month_count",
        "sample_size",
    }
    missing = sorted(required - set(base.columns))
    if missing:
        raise ValueError(f"Professor-facing reliability gate is missing columns: {missing}")
    mask = (
        (base["entity_level"].astype("string") == "neighborhood")
        & base["city_slug"].astype("string").isin(FOUR_CITY_SLUGS)
        & base["neighborhood_slug"].notna()
        & (pd.to_numeric(base["price_month_count"], errors="coerce") >= minimum_months)
        & (pd.to_numeric(base["supply_month_count"], errors="coerce") >= minimum_months)
        & (pd.to_numeric(base["sample_size"], errors="coerce") >= minimum_sample_size)
    )
    return base.loc[mask].copy()


def _consecutive_mom(values: pd.Series, months: pd.Series) -> list[float]:
    numeric = pd.to_numeric(values, errors="coerce")
    parsed = pd.to_datetime(months.astype(str).str[:7] + "-01", errors="coerce")
    work = pd.DataFrame({"month": parsed, "value": numeric}).dropna()
    work = work.loc[work["value"] > 0].sort_values("month")
    if len(work) < 2:
        return []
    output: list[float] = []
    previous_month = None
    previous_value = None
    for row in work.itertuples(index=False):
        current_month = row.month
        current_value = float(row.value)
        if previous_month is not None and previous_value is not None:
            month_gap = (current_month.year - previous_month.year) * 12 + (current_month.month - previous_month.month)
            if month_gap == 1 and previous_value > 0:
                output.append((current_value / previous_value - 1.0) * 100.0)
        previous_month = current_month
        previous_value = current_value
    return output


def _direction_consistency(values: list[float], trend: float) -> float:
    if not values:
        return 0.0
    if trend > 0:
        return float(np.mean(np.asarray(values) > 0))
    if trend < 0:
        return float(np.mean(np.asarray(values) < 0))
    return float(np.mean(np.abs(np.asarray(values)) < 1e-12))


def build_legacy_v1_features(monthly: pd.DataFrame, market_map: pd.DataFrame) -> pd.DataFrame:
    required_map = {
        "city_slug",
        "neighborhood_slug",
        "listing_n",
        "median_asking_price_per_sqm_toman",
        "iqr_asking_price_per_sqm_toman",
        "reliable_flag",
    }
    missing_map = sorted(required_map - set(market_map.columns))
    if missing_map:
        raise ValueError(f"Market-map summary is missing reconciliation columns: {missing_map}")

    map_work = market_map.loc[
        market_map["city_slug"].astype("string").isin(FOUR_CITY_SLUGS)
        & market_map["neighborhood_slug"].notna()
    ].copy()
    map_work["reliable_flag"] = _bool_series(map_work["reliable_flag"])
    map_work = map_work.loc[map_work["reliable_flag"]].copy()
    map_lookup = map_work.set_index(["city_slug", "neighborhood_slug"], drop=False)

    subset = monthly.loc[
        (monthly["entity_level"].astype("string") == "neighborhood")
        & monthly["city_slug"].astype("string").isin(FOUR_CITY_SLUGS)
        & monthly["neighborhood_slug"].notna()
    ].copy()

    rows: list[dict[str, Any]] = []
    for (city, neighborhood), group in subset.groupby(["city_slug", "neighborhood_slug"], dropna=False, sort=False):
        lookup_key = (city, neighborhood)
        if lookup_key not in map_lookup.index:
            continue
        map_row = map_lookup.loc[lookup_key]
        if isinstance(map_row, pd.DataFrame):
            map_row = map_row.iloc[0]

        group = group.sort_values("analysis_month")
        price_group = group.loc[_bool_series(group["price_reliable_flag"])].copy()
        supply_changes = _consecutive_mom(group["deduplicated_listing_count"], group["analysis_month"])
        price_changes = _consecutive_mom(price_group["median_asking_price_per_sqm_toman"], price_group["analysis_month"])
        supply_months = int(pd.to_numeric(group["deduplicated_listing_count"], errors="coerce").gt(0).sum())
        price_months = int(pd.to_numeric(price_group["median_asking_price_per_sqm_toman"], errors="coerce").gt(0).sum())
        if supply_months < LEGACY_MIN_MONTHS or price_months < LEGACY_MIN_MONTHS:
            continue
        if len(supply_changes) < LEGACY_MIN_MOM or len(price_changes) < LEGACY_MIN_MOM:
            continue

        supply_growth = float(np.median(supply_changes))
        price_growth = float(np.median(price_changes))
        supply_consistency = _direction_consistency(supply_changes, supply_growth)
        price_consistency = _direction_consistency(price_changes, price_growth)
        supply_coverage = min(max(len(supply_changes) / max(supply_months - 1, 1), 0.0), 1.0)
        price_coverage = min(max(len(price_changes) / max(price_months - 1, 1), 0.0), 1.0)
        supply_stability = 0.60 * supply_consistency + 0.40 * supply_coverage
        price_stability = 0.60 * price_consistency + 0.40 * price_coverage
        trend_stability = (supply_stability + price_stability) / 2.0

        median_price = pd.to_numeric(pd.Series([map_row["median_asking_price_per_sqm_toman"]]), errors="coerce").iloc[0]
        iqr_price = pd.to_numeric(pd.Series([map_row["iqr_asking_price_per_sqm_toman"]]), errors="coerce").iloc[0]
        listing_n = pd.to_numeric(pd.Series([map_row["listing_n"]]), errors="coerce").iloc[0]
        if not np.isfinite(median_price) or median_price <= 0 or not np.isfinite(iqr_price) or not np.isfinite(listing_n):
            continue

        rows.append(
            {
                "entity_level": "neighborhood",
                "city_slug": city,
                "neighborhood_slug": neighborhood,
                "price_trend_pct_per_month": price_growth,
                "supply_trend_pct_per_month": supply_growth,
                "supply_tightness_pct_per_month": -supply_growth,
                "price_direction_consistency": price_consistency,
                "supply_direction_consistency": supply_consistency,
                "price_month_count": price_months,
                "supply_month_count": supply_months,
                "trend_stability_score": trend_stability,
                "relative_iqr": float(iqr_price / median_price),
                "sample_size": int(listing_n),
            }
        )
    return pd.DataFrame(rows)


def score_legacy_v1(features: pd.DataFrame, hot_q: float, cold_q: float) -> pd.DataFrame:
    out = features.copy()
    if out.empty:
        return out
    out["supply_growth_score"] = _rank_scaled_minus1_plus1(out["supply_trend_pct_per_month"])
    out["price_growth_score"] = _rank_scaled_minus1_plus1(out["price_trend_pct_per_month"])
    out["price_dispersion_score"] = 1.0 - _rank_scaled_zero_one(out["relative_iqr"])
    out["valid_listing_score"] = _rank_scaled_zero_one(out["sample_size"])
    out["evidence_multiplier"] = (
        0.50
        + 0.20 * out["price_dispersion_score"]
        + 0.15 * out["valid_listing_score"]
        + 0.15 * out["trend_stability_score"]
    )
    directional = 0.60 * out["price_growth_score"] + 0.40 * out["supply_growth_score"]
    out["market_temperature_score"] = 100.0 * directional * out["evidence_multiplier"]
    n = len(out)
    if n <= 1:
        out["market_temperature_percentile"] = 0.5
    else:
        out["market_temperature_percentile"] = (out["market_temperature_score"].rank(method="average") - 1.0) / float(n - 1)
    pct = out["market_temperature_percentile"]
    out["market_temperature_label"] = np.select(
        [pct >= hot_q, pct <= cold_q],
        ["HOT", "COLD"],
        default="NEUTRAL",
    )
    out["market_temperature_rank"] = out["market_temperature_score"].rank(method="first", ascending=False)
    return out


def _scenario_table(
    scored: pd.DataFrame,
    *,
    scenario: str,
    score_column: str,
    percentile_column: str,
    label_column: str,
    rank_column: str,
    trend_method: str,
    supply_signal_mode: str,
    ranking_universe: str,
    direction_gate: str,
) -> pd.DataFrame:
    work = scored.loc[
        (scored["entity_level"].astype("string") == "neighborhood")
        & scored["city_slug"].astype("string").isin(FOUR_CITY_SLUGS)
        & scored["neighborhood_slug"].notna()
    ].copy()
    if work.empty:
        return work
    work["scenario"] = scenario
    work["score"] = pd.to_numeric(work[score_column], errors="coerce")
    work["percentile"] = pd.to_numeric(work[percentile_column], errors="coerce")
    work["label"] = work[label_column].astype("string")
    work["rank_in_scenario_universe"] = pd.to_numeric(work[rank_column], errors="coerce")
    work["rank_within_four_city"] = work["score"].rank(method="first", ascending=False)
    work["trend_method"] = trend_method
    work["supply_signal_mode"] = supply_signal_mode
    work["ranking_universe"] = ranking_universe
    work["direction_gate"] = direction_gate
    work["extreme_bucket"] = "OTHER"
    hot_idx = work.loc[work["label"] == "HOT"].nlargest(5, "score").index
    cold_idx = work.loc[work["label"] == "COLD"].nsmallest(5, "score").index
    work.loc[hot_idx, "extreme_bucket"] = "TOP5_HOT"
    work.loc[cold_idx, "extreme_bucket"] = "TOP5_COLD"
    keep = [
        "scenario",
        "city_slug",
        "neighborhood_slug",
        "score",
        "percentile",
        "label",
        "rank_in_scenario_universe",
        "rank_within_four_city",
        "extreme_bucket",
        "price_trend_pct_per_month",
        "supply_trend_pct_per_month",
        "evidence_multiplier",
        "trend_method",
        "supply_signal_mode",
        "ranking_universe",
        "direction_gate",
    ]
    return work[keep].sort_values(["scenario", "rank_within_four_city"])


def build_reconciliation(
    current_features: pd.DataFrame,
    legacy_features: pd.DataFrame,
    *,
    price_weight: float,
    supply_weight: float,
    hot_q: float,
    cold_q: float,
) -> pd.DataFrame:
    scenarios: list[pd.DataFrame] = []

    final_all = _score(
        current_features,
        price_weight,
        supply_weight,
        hot_q,
        cold_q,
        "final_all",
        supply_signal="activity",
        gate_mode="price_only",
    )
    scenarios.append(
        _scenario_table(
            final_all,
            scenario="final_all_city_activity_60_40",
            score_column="temperature_score_final_all",
            percentile_column="temperature_percentile_final_all",
            label_column="temperature_label_final_all",
            rank_column="temperature_rank_final_all",
            trend_method="log_linear",
            supply_signal_mode="positive_listing_growth_as_activity",
            ranking_universe="all_entities_within_level",
            direction_gate="price_only",
        )
    )

    four_features = current_features.loc[
        (current_features["entity_level"].astype("string") == "neighborhood")
        & current_features["city_slug"].astype("string").isin(FOUR_CITY_SLUGS)
        & current_features["neighborhood_slug"].notna()
    ].copy()

    final_four = _score(
        four_features,
        price_weight,
        supply_weight,
        hot_q,
        cold_q,
        "final_four",
        supply_signal="activity",
        gate_mode="price_only",
    )
    scenarios.append(
        _scenario_table(
            final_four,
            scenario="final_four_city_activity_60_40",
            score_column="temperature_score_final_four",
            percentile_column="temperature_percentile_final_four",
            label_column="temperature_label_final_four",
            rank_column="temperature_rank_final_four",
            trend_method="log_linear",
            supply_signal_mode="positive_listing_growth_as_activity",
            ranking_universe="four_city_neighborhoods",
            direction_gate="price_only",
        )
    )

    sensitivity_70_30 = _score(
        current_features,
        0.70,
        0.30,
        hot_q,
        cold_q,
        "activity_70_30",
        supply_signal="activity",
        gate_mode="price_only",
    )
    scenarios.append(
        _scenario_table(
            sensitivity_70_30,
            scenario="sensitivity_all_city_activity_70_30",
            score_column="temperature_score_activity_70_30",
            percentile_column="temperature_percentile_activity_70_30",
            label_column="temperature_label_activity_70_30",
            rank_column="temperature_rank_activity_70_30",
            trend_method="log_linear",
            supply_signal_mode="positive_listing_growth_as_activity",
            ranking_universe="all_entities_within_level",
            direction_gate="price_only",
        )
    )

    superseded = _score(
        four_features,
        price_weight,
        supply_weight,
        hot_q,
        cold_q,
        "superseded_tightness",
        supply_signal="tightness",
        gate_mode="price_and_supply",
    )
    scenarios.append(
        _scenario_table(
            superseded,
            scenario="superseded_four_city_tightness_strict",
            score_column="temperature_score_superseded_tightness",
            percentile_column="temperature_percentile_superseded_tightness",
            label_column="temperature_label_superseded_tightness",
            rank_column="temperature_rank_superseded_tightness",
            trend_method="log_linear",
            supply_signal_mode="negative_supply_growth_as_tightness",
            ranking_universe="four_city_neighborhoods",
            direction_gate="price_and_opposite_supply",
        )
    )

    legacy = score_legacy_v1(legacy_features, hot_q, cold_q)
    if not legacy.empty:
        scenarios.append(
            _scenario_table(
                legacy,
                scenario="legacy_v1_reconstruction",
                score_column="market_temperature_score",
                percentile_column="market_temperature_percentile",
                label_column="market_temperature_label",
                rank_column="market_temperature_rank",
                trend_method="median_consecutive_mom",
                supply_signal_mode="positive_supply_growth_as_activity",
                ranking_universe="four_city_neighborhoods",
                direction_gate="none_percentile_only",
            )
        )

    valid = [frame for frame in scenarios if not frame.empty]
    if not valid:
        return pd.DataFrame()
    return pd.concat(valid, ignore_index=True)


def _top_set(reconciliation: pd.DataFrame, scenario: str, bucket: str) -> set[str]:
    rows = reconciliation.loc[
        (reconciliation["scenario"] == scenario) & (reconciliation["extreme_bucket"] == bucket)
    ]
    return set(rows["city_slug"].astype(str) + "|" + rows["neighborhood_slug"].astype(str))


def _print_reconciliation(reconciliation: pd.DataFrame) -> None:
    print("M3 MARKET TEMPERATURE RECONCILIATION")
    for scenario in reconciliation["scenario"].drop_duplicates().tolist():
        part = reconciliation.loc[reconciliation["scenario"] == scenario]
        hot = part.loc[part["extreme_bucket"] == "TOP5_HOT", ["city_slug", "neighborhood_slug", "score"]]
        cold = part.loc[part["extreme_bucket"] == "TOP5_COLD", ["city_slug", "neighborhood_slug", "score"]]
        hot_text = "; ".join(f"{r.city_slug}/{r.neighborhood_slug}={r.score:+.1f}" for r in hot.itertuples(index=False)) or "none"
        cold_text = "; ".join(f"{r.city_slug}/{r.neighborhood_slug}={r.score:+.1f}" for r in cold.itertuples(index=False)) or "none"
        print(f"  {scenario}")
        print(f"    HOT : {hot_text}")
        print(f"    COLD: {cold_text}")


def run(monthly_path: Path = MONTHLY_MARKET, market_map_path: Path = MARKET_MAP) -> dict[str, Path]:
    monthly_path = monthly_path.resolve()
    market_map_path = market_map_path.resolve()
    for label, path in [("monthly market summary", monthly_path), ("market map summary", market_map_path)]:
        if not path.exists():
            raise FileNotFoundError(f"Required {label} not found: {path}")

    price_weight = float(setting("milestone_3", "market_temperature", "price_weight", default=0.60))
    activity_weight = float(
        setting(
            "milestone_3",
            "market_temperature",
            "listing_activity_weight",
            default=setting("milestone_3", "market_temperature", "supply_tightness_weight", default=0.40),
        )
    )
    hot_q = float(setting("milestone_3", "market_temperature", "hot_percentile", default=0.80))
    cold_q = float(setting("milestone_3", "market_temperature", "cold_percentile", default=0.20))
    min_months = int(setting("milestone_3", "market_temperature", "minimum_months", default=4))
    surface_min_months = int(
        setting("milestone_3", "market_temperature", "professor_facing_minimum_months", default=5)
    )
    surface_min_n = int(
        setting("milestone_3", "market_temperature", "professor_facing_minimum_n", default=100)
    )

    table_dir = OUTPUTS_DIR / "tables" / "milestone_3" / "market_temperature"
    qa_dir = OUTPUTS_DIR / "qa" / "milestone_3" / "market_temperature"
    fig_dir = OUTPUTS_DIR / "figures" / "milestone_3" / "market_temperature"
    table_dir.mkdir(parents=True, exist_ok=True)
    qa_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    show_progress(0, "loading monthly market")

    monthly = pd.read_csv(monthly_path)
    required = {
        "analysis_month",
        "entity_level",
        "city_slug",
        "neighborhood_slug",
        "deduplicated_listing_count",
        "median_asking_price_per_sqm_toman",
        "price_reliable_flag",
        "supply_population",
        "price_population",
    }
    missing = sorted(required - set(monthly.columns))
    if missing:
        raise ValueError(
            "Monthly market summary is missing required columns: "
            f"{missing}. Re-run monthly_market_summary before market_temperature."
        )

    supply_populations = set(monthly["supply_population"].dropna().astype(str).unique())
    price_populations = set(monthly["price_population"].dropna().astype(str).unique())
    population_aligned = supply_populations == {"apartment_sale"} and price_populations == {"apartment_sale"}
    if not population_aligned:
        raise ValueError(
            "Market-temperature population is not aligned. Expected apartment_sale listing activity and apartment_sale price. "
            f"supply={sorted(supply_populations)}, price={sorted(price_populations)}"
        )

    market_map = pd.read_csv(market_map_path)
    show_progress(20, f"loaded {len(monthly):,} monthly rows")

    features = build_features(monthly, min_months)
    if features.empty:
        raise RuntimeError("No market-temperature entities had enough price and listing-activity history.")
    show_progress(48, f"trend features: {len(features):,} entities")

    base = _score(
        features,
        price_weight,
        activity_weight,
        hot_q,
        cold_q,
        "base",
        supply_signal="activity",
        gate_mode="price_only",
    )
    base = base.rename(
        columns={
            "temperature_score_base": "market_temperature_score",
            "temperature_percentile_base": "market_temperature_percentile",
            "temperature_label_base": "market_temperature_label",
            "temperature_rank_base": "market_temperature_rank",
        }
    )
    base["proxy_definition"] = "Listing-Market Temperature Proxy; asking-price trend plus listing-activity trend; not liquidity or absorption"
    base["supply_population"] = "apartment_sale"
    base["price_population"] = "apartment_sale"
    base["listing_activity_interpretation"] = "platform listing activity; not physical inventory or absorption"
    base["analysis_version"] = VERSION

    sensitivity_rows: list[pd.DataFrame] = []
    for label, pw, aw in [("primary_60_40", price_weight, activity_weight), ("price_70_activity_30", 0.70, 0.30)]:
        temp = _score(
            features,
            pw,
            aw,
            hot_q,
            cold_q,
            label,
            supply_signal="activity",
            gate_mode="price_only",
        )
        keep = [
            "entity_level",
            "city_slug",
            "neighborhood_slug",
            f"temperature_score_{label}",
            f"temperature_percentile_{label}",
            f"temperature_label_{label}",
        ]
        part = temp[keep].copy()
        part["scenario"] = label
        part["price_weight"] = pw
        part["listing_activity_weight"] = aw
        part = part.rename(
            columns={
                f"temperature_score_{label}": "market_temperature_score",
                f"temperature_percentile_{label}": "market_temperature_percentile",
                f"temperature_label_{label}": "market_temperature_label",
            }
        )
        sensitivity_rows.append(part)
    sensitivity = pd.concat(sensitivity_rows, ignore_index=True)

    legacy_features = build_legacy_v1_features(monthly, market_map)
    reconciliation = build_reconciliation(
        features,
        legacy_features,
        price_weight=price_weight,
        supply_weight=activity_weight,
        hot_q=hot_q,
        cold_q=cold_q,
    )
    if reconciliation.empty:
        raise RuntimeError("Market-temperature audit produced no four-city neighborhood rows.")
    show_progress(75, "final and sensitivity scenarios complete")

    public_summary = base.drop(columns=["supply_tightness_pct_per_month"], errors="ignore").copy()
    base_pl = pl.from_pandas(public_summary)
    sensitivity_pl = pl.from_pandas(sensitivity)
    reconciliation_pl = pl.from_pandas(reconciliation)

    primary_hot = _top_set(reconciliation, "final_all_city_activity_60_40", "TOP5_HOT")
    primary_cold = _top_set(reconciliation, "final_all_city_activity_60_40", "TOP5_COLD")
    sensitivity_hot = _top_set(reconciliation, "sensitivity_all_city_activity_70_30", "TOP5_HOT")
    sensitivity_cold = _top_set(reconciliation, "sensitivity_all_city_activity_70_30", "TOP5_COLD")
    legacy_hot = _top_set(reconciliation, "legacy_v1_reconstruction", "TOP5_HOT")
    legacy_cold = _top_set(reconciliation, "legacy_v1_reconstruction", "TOP5_COLD")
    hot_sensitivity_overlap = len(primary_hot & sensitivity_hot)
    cold_sensitivity_overlap = len(primary_cold & sensitivity_cold)
    hot_legacy_overlap = len(primary_hot & legacy_hot)
    cold_legacy_overlap = len(primary_cold & legacy_cold)

    surfaced_neighborhoods = _professor_facing_neighborhoods(
        base,
        minimum_months=surface_min_months,
        minimum_sample_size=surface_min_n,
    )
    surfaced_hot_pool = surfaced_neighborhoods.loc[
        surfaced_neighborhoods["market_temperature_label"] == "HOT"
    ].copy()
    surfaced_cold_pool = surfaced_neighborhoods.loc[
        surfaced_neighborhoods["market_temperature_label"] == "COLD"
    ].copy()
    surfaced_hot = (
        surfaced_hot_pool.nlargest(min(5, len(surfaced_hot_pool)), "market_temperature_score")
        if not surfaced_hot_pool.empty
        else surfaced_hot_pool
    )
    surfaced_cold = (
        surfaced_cold_pool.nsmallest(min(5, len(surfaced_cold_pool)), "market_temperature_score")
        if not surfaced_cold_pool.empty
        else surfaced_cold_pool
    )
    surfaced_extremes = pd.concat([surfaced_cold, surfaced_hot], ignore_index=True)
    reliability_violations = int(
        (
            (pd.to_numeric(surfaced_extremes.get("price_month_count"), errors="coerce") < surface_min_months)
            | (pd.to_numeric(surfaced_extremes.get("supply_month_count"), errors="coerce") < surface_min_months)
            | (pd.to_numeric(surfaced_extremes.get("sample_size"), errors="coerce") < surface_min_n)
        ).fillna(True).sum()
    ) if not surfaced_extremes.empty else 0

    checks: list[Check] = [
        make_check("temperature_entities_nonempty", "temperature", len(base), ">0", len(base) > 0),
        make_check(
            "weights_sum_to_one",
            "temperature",
            price_weight + activity_weight,
            1.0,
            abs(price_weight + activity_weight - 1.0) < 1e-9,
        ),
        make_check("temperature_population_alignment", "temperature", population_aligned, True, population_aligned),
        make_check(
            "hot_price_direction_gate",
            "temperature",
            int(((base["market_temperature_label"] == "HOT") & (base["price_trend_pct_per_month"] <= 0)).sum()),
            0,
            not ((base["market_temperature_label"] == "HOT") & (base["price_trend_pct_per_month"] <= 0)).any(),
        ),
        make_check(
            "cold_price_direction_gate",
            "temperature",
            int(((base["market_temperature_label"] == "COLD") & (base["price_trend_pct_per_month"] >= 0)).sum()),
            0,
            not ((base["market_temperature_label"] == "COLD") & (base["price_trend_pct_per_month"] >= 0)).any(),
        ),
        make_check(
            "activity_weight_sensitivity_top5_hot",
            "sensitivity",
            f"{hot_sensitivity_overlap}/5",
            ">=3/5",
            hot_sensitivity_overlap >= 3,
            critical=False,
            review_on_fail=True,
            notes="Primary 60/40 versus price-heavier 70/30 top-5 HOT overlap.",
        ),
        make_check(
            "activity_weight_sensitivity_top5_cold",
            "sensitivity",
            f"{cold_sensitivity_overlap}/5",
            ">=3/5",
            cold_sensitivity_overlap >= 3,
            critical=False,
            review_on_fail=True,
            notes="Primary 60/40 versus price-heavier 70/30 top-5 COLD overlap.",
        ),
        make_check("legacy_reconstruction_nonempty", "reconciliation", len(legacy_features), ">0", len(legacy_features) > 0),
        make_check(
            "professor_facing_reliability_gate",
            "presentation",
            reliability_violations,
            0,
            reliability_violations == 0,
            critical=False,
            review_on_fail=True,
            notes=f"Surfaced top-5 extrema require >= {surface_min_months} reliable price/activity months and N >= {surface_min_n}.",
        ),
        make_check(
            "professor_facing_hot_count",
            "presentation",
            len(surfaced_hot),
            5,
            len(surfaced_hot) >= 5,
            critical=False,
            review_on_fail=True,
            notes="Professor-facing chart should surface five reliable HOT neighborhoods when available.",
        ),
        make_check(
            "professor_facing_cold_count",
            "presentation",
            len(surfaced_cold),
            5,
            len(surfaced_cold) >= 5,
            critical=False,
            review_on_fail=True,
            notes="Professor-facing chart should surface five reliable COLD neighborhoods when available.",
        ),
        make_check(
            "deprecated_tightness_not_surfaced",
            "output_contract",
            "supply_tightness_pct_per_month" in public_summary.columns,
            False,
            "supply_tightness_pct_per_month" not in public_summary.columns,
            notes="The superseded tightness transform is retained only inside QA reconciliation, not the surfaced summary.",
        ),
    ]

    figure_path = fig_dir / "city_market_temperature_ranking.png"
    if not surfaced_neighborhoods.empty:
        hot_pool = surfaced_hot_pool
        cold_pool = surfaced_cold_pool
        hot = surfaced_hot
        cold = surfaced_cold
        city_plot = pd.concat([cold, hot], ignore_index=True).drop_duplicates(subset=["city_slug", "neighborhood_slug"]).sort_values("market_temperature_score")
        if not city_plot.empty:
            scores = city_plot["market_temperature_score"].to_numpy(dtype=float)
            labels = [
                f"{_display_slug(c)} / {_display_slug(n)}"
                for c, n in zip(city_plot["city_slug"].astype(str), city_plot["neighborhood_slug"].astype(str))
            ]
            colors = ["#2F6B9A" if label == "COLD" else "#C96B2C" for label in city_plot["market_temperature_label"].astype(str)]

            fig, ax = plt.subplots(figsize=(12.0, 7.2))
            bars = ax.barh(range(len(city_plot)), scores, color=colors)
            ax.set_yticks(range(len(city_plot)), labels)
            ax.axvline(0, linewidth=1.2, color="#666666")
            ax.set_xlabel("Listing-Market Temperature Proxy score")
            ax.grid(axis="x", alpha=0.18, linewidth=0.8)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
            fig.suptitle(
                "Neighborhood Market Temperature - 5 Hottest and 5 Coldest",
                x=0.5,
                y=0.965,
                fontsize=15.5,
                fontweight="bold",
            )
            fig.text(
                0.5,
                0.922,
                (
                    "Apartment-sale asking price + listing activity | all-city ranking; four-city reliable view "
                    f"(>= {surface_min_months} months, N >= {surface_min_n}) | "
                    f"HOT eligible={len(hot_pool):,}, COLD eligible={len(cold_pool):,}"
                ),
                ha="center",
                fontsize=9.1,
                color="#555555",
            )
            label_offset = max(abs(scores).max(), 1.0) * 0.025
            for bar, value, n_value in zip(bars, scores, city_plot["sample_size"].fillna(0).astype(int)):
                ax.text(
                    label_offset if value >= 0 else -label_offset,
                    bar.get_y() + bar.get_height() / 2,
                    f"{value:+.1f}  |  N={n_value:,}",
                    va="center",
                    ha="left" if value >= 0 else "right",
                    fontsize=8.4,
                    color="white",
                    fontweight="semibold",
                )
            fig.text(
                0.5,
                0.025,
                "Temperature is a relative listing-market proxy, not liquidity, absorption, or physical inventory. HOT/COLD also require the observed asking-price trend to have the corresponding sign.",
                ha="center",
                fontsize=8.2,
                color="#555555",
            )
            fig.tight_layout(rect=(0.04, 0.075, 0.99, 0.885))
            fig.savefig(figure_path, dpi=300, bbox_inches="tight", facecolor="white")
            plt.close(fig)

    summary_path = table_dir / "market_temperature_summary.csv"
    sensitivity_path = qa_dir / "market_temperature_sensitivity.csv"
    reconciliation_path = qa_dir / "market_temperature_reconciliation.csv"
    checks_path = qa_dir / "market_temperature_checks.csv"
    manifest_path = qa_dir / "market_temperature_manifest.json"
    atomic_write_csv(base_pl, summary_path)
    atomic_write_csv(sensitivity_pl, sensitivity_path)
    atomic_write_csv(reconciliation_pl, reconciliation_path)
    atomic_write_csv(checks_frame(checks), checks_path)
    status = summarize_checks(checks)
    freeze_authorized = bool(status.get("ready") and status.get("overall_status") == "PASS")

    atomic_write_json(
        {
            "version": VERSION,
            "status": status,
            "freeze_authorized": freeze_authorized,
            "methodology_status": "frozen_final" if freeze_authorized else "final_review_required",
            "inputs": {
                "monthly_market": relative_to_project(monthly_path),
                "market_map": relative_to_project(market_map_path),
            },
            "population": {
                "listing_activity": "apartment_sale",
                "price": "apartment_sale",
                "aligned": population_aligned,
            },
            "definition": {
                "name": "Listing-Market Temperature Proxy",
                "primary_weights": {"asking_price_trend": price_weight, "listing_activity_trend": activity_weight},
                "primary_trend_estimator": "log-linear monthly slope using all usable months",
                "listing_activity_semantics": "positive growth means higher platform listing activity; it is not interpreted as physical inventory, tightness, liquidity, or absorption",
                "hot_percentile": hot_q,
                "cold_percentile": cold_q,
                "minimum_months": min_months,
                "professor_facing_reliability_gate": {
                    "minimum_reliable_price_months": surface_min_months,
                    "minimum_listing_activity_months": surface_min_months,
                    "minimum_sample_size": surface_min_n,
                    "scope": "presentation only; canonical all-city ranking and labels remain unchanged",
                },
                "direction_gate": "HOT requires positive asking-price trend; COLD requires negative asking-price trend; listing-activity sign is not a hard gate",
                "evidence_multiplier": "0.65 + 0.20*month_coverage + 0.15*trend_direction_stability; reliable-price months are enforced upstream",
                "ranking_universe": "all eligible entities within each entity_level; professor-facing figure shows only reliable Tehran, Mashhad, Karaj, and Isfahan neighborhoods",
                "public_summary_policy": "superseded supply_tightness_pct_per_month is excluded from the surfaced summary and retained only in QA reconciliation logic",
            },
            "sensitivity": {
                "weight_scenario": {"asking_price_trend": 0.70, "listing_activity_trend": 0.30},
                "top5_hot_overlap_with_primary": hot_sensitivity_overlap,
                "top5_cold_overlap_with_primary": cold_sensitivity_overlap,
                "stability_gate": ">=3 of 5 for HOT and COLD",
            },
            "reconciliation": {
                "scenarios": reconciliation["scenario"].drop_duplicates().tolist(),
                "primary_vs_legacy_top5_hot_overlap": hot_legacy_overlap,
                "primary_vs_legacy_top5_cold_overlap": cold_legacy_overlap,
                "superseded_tightness_method_retained_for_qa_only": True,
                "legacy_median_mom_method_retained_for_qa_only": True,
            },
            "outputs": {
                "summary": relative_to_project(summary_path),
                "sensitivity": relative_to_project(sensitivity_path),
                "reconciliation": relative_to_project(reconciliation_path),
                "checks": relative_to_project(checks_path),
                "figure": relative_to_project(figure_path) if figure_path.exists() else None,
            },
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        },
        manifest_path,
    )

    _print_reconciliation(reconciliation)
    print(
        "M3 MARKET TEMPERATURE SENSITIVITY "
        f"HOT overlap primary-vs-70/30={hot_sensitivity_overlap}/5 | "
        f"COLD overlap={cold_sensitivity_overlap}/5"
    )
    print(
        "M3 MARKET TEMPERATURE SURFACED RELIABILITY "
        f"HOT={len(surfaced_hot)}/5 | COLD={len(surfaced_cold)}/5 | "
        f"minimum_months={surface_min_months} | minimum_n={surface_min_n}"
    )
    print(f"M3 MARKET TEMPERATURE FREEZE_AUTHORIZED={str(freeze_authorized).lower()}")
    show_progress(100, f"complete in {time.perf_counter()-started:.1f}s", final=True)
    return {
        "summary": summary_path,
        "sensitivity": sensitivity_path,
        "reconciliation": reconciliation_path,
        "figure": figure_path,
        "checks": checks_path,
        "manifest": manifest_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the frozen-final all-city/neighborhood Listing-Market Temperature Proxy.")
    parser.add_argument("--monthly-market", type=Path, default=MONTHLY_MARKET)
    parser.add_argument("--market-map", type=Path, default=MARKET_MAP)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run(args.monthly_market, args.market_map)
    print("M3 MARKET TEMPERATURE FINAL RUN COMPLETED")
    for name, path in outputs.items():
        print(f"{name}: {relative_to_project(path)}")


if __name__ == "__main__":
    main()
