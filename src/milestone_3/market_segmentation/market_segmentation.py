from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import RobustScaler

from src.common.config import setting
from src.common.io_utils import atomic_sink_parquet, atomic_write_csv, atomic_write_json
from src.common.paths import OUTPUTS_DIR, relative_to_project
from src.common.validation import Check, checks_frame, make_check, summarize_checks

VERSION = "m3-market-segmentation-v1.5-layout-semantic-labels"
FEATURE_TABLE = OUTPUTS_DIR / "model_artifacts" / "milestone_3" / "price_drivers" / "analysis_ready_features.parquet"
TEXT_TABLE = OUTPUTS_DIR / "model_artifacts" / "milestone_3" / "text_analysis" / "text_features.parquet"
TARGET = "sale_price_per_sqm_final_toman"
PROGRESS_WIDTH = 30
CORE_FEATURES = ["relative_price_log", "log_area", "rooms_count_num", "building_age_years", "construction_year_before_1370_flag"]
OPTIONAL_FEATURES = ["floor_num", "has_elevator_bool", "has_parking_bool", "has_warehouse_bool", "has_balcony_bool"]
BINARY_FEATURES = {"construction_year_before_1370_flag", "has_elevator_bool", "has_parking_bool", "has_warehouse_bool", "has_balcony_bool"}
PRESENTATION_MAX_CLUSTERS = 5
AMENITY_FEATURES = ["has_elevator_bool", "has_parking_bool", "has_warehouse_bool", "has_balcony_bool"]
FALLBACK_SCORE_WEIGHTS = {"relative_price_rank": 0.55, "area_rank": 0.20, "amenity_score": 0.15, "newness_score": 0.07, "rooms_rank": 0.03}
FALLBACK_ECONOMIC_QUANTILE = 0.30
FALLBACK_LUXURY_QUANTILE = 0.80
LUXURY_NEWER_MAX_MEDIAN_AGE_YEARS = 5.0
LUXURY_AMENITY_RICH_MIN_RATE = 0.80
PRIMARY_PRESENTATION_IDS = ("SEG01", "SEG02", "SEG03", "SEG04", "SEG05")
DOMAIN_SEGMENTS = {
    "land": ("SEG04", "\u0632\u0645\u06cc\u0646/\u0645\u0644\u06a9 \u0633\u0631\u0645\u0627\u06cc\u0647\u200c\u0627\u06cc"),
    "commercial": ("SEG05", "\u0645\u0644\u06a9 \u062a\u062c\u0627\u0631\u06cc/\u0627\u062f\u0627\u0631\u06cc"),
}


@dataclass
class TransformState:
    neighborhood_reference: pd.DataFrame
    city_reference: pd.DataFrame
    global_reference: float
    features: list[str]
    fill_values: dict[str, float]
    missing_indicators: list[str]
    scaler: RobustScaler


PROFILE_LABELS = {
    "sale_price_per_sqm_final_toman": "Asking PSM",
    "primary_area_sqm": "Area",
    "rooms_count_num": "Rooms",
    "building_age_years": "Building age",
    "floor_num": "Floor",
    "has_elevator_bool": "Elevator rate",
    "has_parking_bool": "Parking rate",
    "has_warehouse_bool": "Storage rate",
    "has_balcony_bool": "Balcony rate",
}


def _profile_label(column: str) -> str:
    base = column.replace("median_", "").replace("rate_", "")
    return PROFILE_LABELS.get(base, base.replace("_", " ").title())


def _segment_label_fa(name: object) -> str:
    raw = str(name or "").strip()
    if any("\u0600" <= ch <= "\u06FF" for ch in raw):
        return raw
    text = raw.lower()
    if "premium" in text or "luxury" in text:
        base = "ممتاز نسبی"
    elif "value" in text or "economic" in text:
        base = "اقتصادی نسبی"
    elif "core" in text or "mid" in text:
        base = "میان‌رده"
    else:
        base = "سگمنت"
    if "spacious" in text:
        base += " | خانوادگی بزرگ‌متراژ"
    elif "compact" in text:
        base += " | کم‌متراژ شهری"
    return base


def _presentation_segments(profile: pd.DataFrame, maximum: int = PRESENTATION_MAX_CLUSTERS) -> pd.DataFrame:
    ids = set(profile.get("segment_id", pd.Series(dtype="string")).astype(str))
    if {"SEG01", "SEG02", "SEG03"}.issubset(ids):
        preferred = profile.loc[profile["segment_id"].astype(str).isin(PRIMARY_PRESENTATION_IDS)].copy()
        order = {segment_id: idx for idx, segment_id in enumerate(PRIMARY_PRESENTATION_IDS)}
        preferred["_presentation_order"] = preferred["segment_id"].astype(str).map(order)
        preferred = preferred.sort_values("_presentation_order").drop(columns="_presentation_order")
        preferred["segment_label_fa"] = preferred["segment_name"].astype(str)
        return preferred.head(maximum).copy()

    view = profile.loc[
        profile["dominant_property_family"].astype(str).eq("apartment")
        & ~profile["segment_method"].astype(str).eq("compatible_domain_segment")
    ].copy()
    if view.empty:
        view = profile.copy()
    view = view.sort_values(["listing_n", "median_sale_price_per_sqm_final_toman"], ascending=[False, True]).head(maximum).copy()
    raw_labels = [_segment_label_fa(v) for v in view["segment_name"]]
    seen: dict[str, int] = {}
    display: list[str] = []
    persian_digits = "\u06f0\u06f1\u06f2\u06f3\u06f4\u06f5\u06f6\u06f7\u06f8\u06f9"
    for label in raw_labels:
        seen[label] = seen.get(label, 0) + 1
        count = seen[label]
        display.append(label if count == 1 else f"{label} | \u06af\u0631\u0648\u0647 {persian_digits[count] if count < 10 else count}")
    view["segment_label_fa"] = display
    return view


def show_progress(percent: int, label: str, *, final: bool = False) -> None:
    pct = max(0, min(100, int(percent)))
    filled = int(PROGRESS_WIDTH * pct / 100)
    bar = "#" * filled + "-" * (PROGRESS_WIDTH - filled)
    print(
        f"\rM3 segments [{bar}] {pct:3d}% complete | {100-pct:3d}% remaining | {label[:42]:42s}",
        end="\n" if final else "", flush=True,
    )


def _sample(frame: pd.DataFrame, maximum: int, seed: int) -> pd.DataFrame:
    if len(frame) <= maximum:
        return frame.copy()
    return frame.sample(n=maximum, random_state=seed)


def fit_reference(frame: pd.DataFrame, min_n: int) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    valid = frame.loc[pd.to_numeric(frame[TARGET], errors="coerce").gt(0)].copy()
    global_ref = float(pd.to_numeric(valid[TARGET], errors="coerce").median())
    city = valid.groupby("city_slug", dropna=False)[TARGET].agg(["median", "size"]).reset_index().rename(columns={"median": "city_ref", "size": "city_ref_n"})
    neighborhood = valid.groupby(["city_slug", "neighborhood_slug"], dropna=False)[TARGET].agg(["median", "size"]).reset_index().rename(columns={"median": "neighborhood_ref", "size": "neighborhood_ref_n"})
    neighborhood.loc[neighborhood["neighborhood_ref_n"] < min_n, "neighborhood_ref"] = np.nan
    return neighborhood, city, global_ref


def apply_reference(frame: pd.DataFrame, neighborhood: pd.DataFrame, city: pd.DataFrame, global_ref: float) -> pd.DataFrame:
    work = frame.copy()
    work = work.merge(neighborhood[["city_slug", "neighborhood_slug", "neighborhood_ref"]], on=["city_slug", "neighborhood_slug"], how="left")
    work = work.merge(city[["city_slug", "city_ref"]], on="city_slug", how="left")
    work["local_reference_psm"] = work["neighborhood_ref"].fillna(work["city_ref"]).fillna(global_ref)
    target = pd.to_numeric(work[TARGET], errors="coerce")
    work["relative_price_log"] = np.log(target / work["local_reference_psm"])
    area = pd.to_numeric(work["primary_area_sqm"], errors="coerce")
    work["log_area"] = np.log(area.where(area > 0))
    return work.drop(columns=["neighborhood_ref", "city_ref"], errors="ignore")


def apply_compatible_reference(frame: pd.DataFrame, min_n: int) -> pd.DataFrame:
    """Attach a local price index within comparable property family for presentation only."""
    work = frame.copy()
    target = pd.to_numeric(work[TARGET], errors="coerce")
    valid = work.loc[target.gt(0)].copy()
    valid[TARGET] = pd.to_numeric(valid[TARGET], errors="coerce")

    family_global = (
        valid.groupby("property_family", dropna=False)[TARGET]
        .median().rename("family_global_ref").reset_index()
    )
    city_family = (
        valid.groupby(["property_family", "city_slug"], dropna=False)[TARGET]
        .agg(["median", "size"]).reset_index()
        .rename(columns={"median": "city_family_ref", "size": "city_family_n"})
    )
    neighborhood_family = (
        valid.groupby(["property_family", "city_slug", "neighborhood_slug"], dropna=False)[TARGET]
        .agg(["median", "size"]).reset_index()
        .rename(columns={"median": "neighborhood_family_ref", "size": "neighborhood_family_n"})
    )
    neighborhood_family.loc[neighborhood_family["neighborhood_family_n"] < min_n, "neighborhood_family_ref"] = np.nan
    city_family.loc[city_family["city_family_n"] < min_n, "city_family_ref"] = np.nan

    global_ref = float(valid[TARGET].median())
    work = work.merge(neighborhood_family[["property_family", "city_slug", "neighborhood_slug", "neighborhood_family_ref"]], on=["property_family", "city_slug", "neighborhood_slug"], how="left")
    work = work.merge(city_family[["property_family", "city_slug", "city_family_ref"]], on=["property_family", "city_slug"], how="left")
    work = work.merge(family_global, on="property_family", how="left")
    work["compatible_local_reference_psm"] = (
        work["neighborhood_family_ref"]
        .fillna(work["city_family_ref"])
        .fillna(work["family_global_ref"])
        .fillna(global_ref)
    )
    target = pd.to_numeric(work[TARGET], errors="coerce")
    work["normalized_price_index"] = target / work["compatible_local_reference_psm"] * 100.0
    return work.drop(columns=["neighborhood_family_ref", "city_family_ref", "family_global_ref"], errors="ignore")


def fit_transform_state(frame: pd.DataFrame, requested_features: list[str], min_n: int, min_optional_coverage: float) -> tuple[np.ndarray, pd.DataFrame, TransformState]:
    neighborhood, city, global_ref = fit_reference(frame, min_n)
    work = apply_reference(frame, neighborhood, city, global_ref)
    features: list[str] = []
    for feature in requested_features:
        if feature not in work.columns:
            continue
        coverage = float(work[feature].notna().mean())
        if feature in OPTIONAL_FEATURES and coverage < min_optional_coverage:
            continue
        features.append(feature)
    if not set(CORE_FEATURES[:4]).issubset(features):
        raise ValueError(f"Core segmentation features are unavailable after feature screening: {features}")

    fill_values: dict[str, float] = {}
    missing_indicators: list[str] = []
    matrix = pd.DataFrame(index=work.index)
    for feature in features:
        series = pd.to_numeric(work[feature], errors="coerce")
        if feature in BINARY_FEATURES:
            fill = float(series.mean()) if series.notna().any() else 0.5
        else:
            fill = float(series.median()) if series.notna().any() else 0.0
        fill_values[feature] = fill
        missing_rate = float(series.isna().mean())
        matrix[feature] = series.fillna(fill)
        if 0.05 <= missing_rate <= 0.50:
            indicator = f"{feature}__missing"
            missing_indicators.append(indicator)
            matrix[indicator] = series.isna().astype(float)
    scaler = RobustScaler()
    values = scaler.fit_transform(matrix.to_numpy(dtype=float))
    state = TransformState(neighborhood, city, global_ref, features, fill_values, missing_indicators, scaler)
    return values, work, state


def transform(frame: pd.DataFrame, state: TransformState) -> tuple[np.ndarray, pd.DataFrame]:
    work = apply_reference(frame, state.neighborhood_reference, state.city_reference, state.global_reference)
    matrix = pd.DataFrame(index=work.index)
    for feature in state.features:
        series = pd.to_numeric(work.get(feature), errors="coerce")
        matrix[feature] = series.fillna(state.fill_values[feature])
        indicator = f"{feature}__missing"
        if indicator in state.missing_indicators:
            matrix[indicator] = series.isna().astype(float)
    ordered = state.features + state.missing_indicators
    matrix = matrix.reindex(columns=ordered)
    return state.scaler.transform(matrix.to_numpy(dtype=float)), work


def stability_ari(frame: pd.DataFrame, requested_features: list[str], k: int, min_n: int, min_optional_coverage: float, train_sample: int, eval_sample: int, runs: int, seed: int) -> float:
    evaluation = _sample(frame, eval_sample, seed + 900)
    label_runs: list[np.ndarray] = []
    for run in range(runs):
        training = _sample(frame, train_sample, seed + 100 + run)
        x_train, _, state = fit_transform_state(training, requested_features, min_n, min_optional_coverage)
        model = MiniBatchKMeans(n_clusters=k, random_state=seed + run, batch_size=4096, n_init=10)
        model.fit(x_train)
        x_eval, _ = transform(evaluation, state)
        label_runs.append(model.predict(x_eval))
    scores = [adjusted_rand_score(a, b) for a, b in itertools.combinations(label_runs, 2)]
    return float(np.mean(scores)) if scores else 1.0


def evaluate_specification(frame: pd.DataFrame, name: str, features: list[str], params: dict[str, Any], seed: int) -> tuple[pd.DataFrame, int | None]:
    selection = _sample(frame, params["selection_sample"], seed)
    x, _, state = fit_transform_state(selection, features, params["reference_min_n"], params["min_optional_coverage"])
    metric_idx = np.arange(len(x))
    if len(metric_idx) > params["metric_sample"]:
        rng = np.random.default_rng(seed)
        metric_idx = rng.choice(metric_idx, size=params["metric_sample"], replace=False)
    x_metric = x[metric_idx]
    rows: list[dict[str, Any]] = []
    preliminary: list[int] = []
    fitted: dict[int, MiniBatchKMeans] = {}
    for k in range(params["k_min"], params["k_max"] + 1):
        model = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=4096, n_init=10)
        labels = model.fit_predict(x)
        fitted[k] = model
        shares = np.bincount(labels, minlength=k) / len(labels)
        metric_labels = labels[metric_idx]
        silhouette = float(silhouette_score(x_metric, metric_labels)) if len(np.unique(metric_labels)) > 1 else np.nan
        ch = float(calinski_harabasz_score(x_metric, metric_labels)) if len(np.unique(metric_labels)) > 1 else np.nan
        db = float(davies_bouldin_score(x_metric, metric_labels)) if len(np.unique(metric_labels)) > 1 else np.nan
        pre_pass = (
            np.isfinite(silhouette) and silhouette >= params["min_silhouette"] and
            np.isfinite(ch) and ch >= params["min_calinski_harabasz"] and
            np.isfinite(db) and db <= params["max_davies_bouldin"] and
            float(shares.min()) >= params["min_cluster_share"]
        )
        if pre_pass:
            preliminary.append(k)
        rows.append({
            "specification": name, "k": k, "silhouette": silhouette, "calinski_harabasz": ch,
            "davies_bouldin": db, "minimum_cluster_share": float(shares.min()),
            "preliminary_gate_pass": pre_pass, "stability_ari": np.nan, "final_gate_pass": False,
        })

    for k in preliminary:
        ari = stability_ari(
            frame, features, k, params["reference_min_n"], params["min_optional_coverage"],
            params["stability_train_sample"], params["stability_eval_sample"], params["stability_runs"], seed + k,
        )
        for row in rows:
            if row["k"] == k:
                row["stability_ari"] = ari
                row["final_gate_pass"] = ari >= params["min_stability_ari"]
                break
    diagnostics = pd.DataFrame(rows)
    passing = diagnostics.loc[diagnostics["final_gate_pass"]].sort_values(["silhouette", "stability_ari"], ascending=False)
    if passing.empty:
        return diagnostics, None
    selected_k = int(passing.iloc[0]["k"])
    return diagnostics, selected_k


def _optional_numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    """Return a numeric Series aligned to frame.index for an optional feature.

    The descriptive fallback is required to degrade gracefully when optional profile
    features (for example age, rooms, or amenities) are unavailable. Missing optional
    features therefore contribute neutral/missing information rather than failing the
    typology.
    """
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _amenity_score_series(frame: pd.DataFrame) -> pd.Series:
    parts: list[pd.Series] = []
    for col in AMENITY_FEATURES:
        if col not in frame.columns:
            continue
        values = pd.to_numeric(frame[col], errors="coerce")
        if not values.notna().any():
            continue
        fill = float(values.mean())
        parts.append(values.fillna(fill).clip(0, 1))
    if not parts:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.concat(parts, axis=1).mean(axis=1)


def _semantic_name_components(frame: pd.DataFrame) -> dict[str, float]:
    area = _optional_numeric_series(frame, "primary_area_sqm")
    age = _optional_numeric_series(frame, "building_age_years")
    rooms = _optional_numeric_series(frame, "rooms_count_num")
    relative = _optional_numeric_series(frame, "relative_price_log")
    amenities = _amenity_score_series(frame)
    return {
        "median_area": float(area.median()) if area.notna().any() else np.nan,
        "median_age": float(age.median()) if age.notna().any() else np.nan,
        "median_rooms": float(rooms.median()) if rooms.notna().any() else np.nan,
        "median_relative_price": float(relative.median()) if relative.notna().any() else np.nan,
        "amenity_score": float(amenities.mean()) if amenities.notna().any() else np.nan,
    }


def _archetype(stats: dict[str, float], refs: dict[str, float]) -> str:
    area = stats["median_area"]
    age = stats["median_age"]
    amenities = stats["amenity_score"]
    rooms = stats["median_rooms"]
    if np.isfinite(age) and np.isfinite(amenities) and age <= refs["age_q35"] and amenities >= refs["amenity_q65"]:
        return "نوساز/کم‌سن امکانات‌دار"
    if np.isfinite(area) and area >= refs["area_q70"]:
        return "خانوادگی بزرگ‌متراژ"
    if np.isfinite(area) and area <= refs["area_q40"]:
        return "کم‌متراژ شهری"
    if np.isfinite(amenities) and amenities >= refs["amenity_q65"]:
        return "امکانات‌دار"
    if np.isfinite(rooms) and rooms >= refs["rooms_median"]:
        return "خانوادگی"
    return "میان‌متراژ"


def _segment_names(apartment: pd.DataFrame, labels: np.ndarray) -> dict[int, str]:
    work = apartment.copy()
    work["cluster"] = labels
    amenity_rows = _amenity_score_series(work)
    area_all = _optional_numeric_series(work, "primary_area_sqm")
    age_all = _optional_numeric_series(work, "building_age_years")
    rooms_all = _optional_numeric_series(work, "rooms_count_num")
    refs = {
        "area_q40": float(area_all.quantile(0.40)),
        "area_q70": float(area_all.quantile(0.70)),
        "age_q35": float(age_all.quantile(0.35)) if age_all.notna().any() else np.nan,
        "amenity_q65": float(amenity_rows.quantile(0.65)) if amenity_rows.notna().any() else np.nan,
        "rooms_median": float(rooms_all.median()) if rooms_all.notna().any() else np.nan,
    }
    cluster_stats = {int(cluster): _semantic_name_components(group) for cluster, group in work.groupby("cluster")}
    ordered = sorted(cluster_stats, key=lambda c: (cluster_stats[c]["median_relative_price"] if np.isfinite(cluster_stats[c]["median_relative_price"]) else 0.0))
    names: dict[int, str] = {}
    for rank, cluster in enumerate(ordered):
        stats = cluster_stats[cluster]
        archetype = _archetype(stats, refs)
        if rank == 0:
            price_tier = "اقتصادی نسبی"
        elif rank == len(ordered) - 1:
            price_tier = "ممتاز نسبی"
        else:
            price_tier = "میان‌رده"
        names[cluster] = f"{price_tier} | {archetype}"
    # Preserve semantic labels while guaranteeing uniqueness when two middle clusters share the same archetype.
    reverse: dict[str, list[int]] = {}
    for cluster, name in names.items():
        reverse.setdefault(name, []).append(cluster)
    for name, clusters in reverse.items():
        if len(clusters) <= 1:
            continue
        ordered_clusters = sorted(clusters, key=lambda c: cluster_stats[c]["median_relative_price"])
        qualifiers = ["کم‌قیمت‌تر", "میانه", "گران‌تر", "بزرگ‌تر", "امکانات‌دارتر"]
        for idx, cluster in enumerate(ordered_clusters):
            names[cluster] = f"{name} | {qualifiers[min(idx, len(qualifiers)-1)]}"
    return names


def _fallback_score(apartment: pd.DataFrame, min_n: int) -> tuple[pd.DataFrame, pd.Series]:
    neighborhood, city, global_ref = fit_reference(apartment, min_n)
    work = apply_reference(apartment, neighborhood, city, global_ref)
    relative = _optional_numeric_series(work, "relative_price_log")
    area = np.log1p(_optional_numeric_series(work, "primary_area_sqm").clip(lower=0))
    age = _optional_numeric_series(work, "building_age_years")
    rooms = _optional_numeric_series(work, "rooms_count_num")
    amenities = _amenity_score_series(work)

    def rank(series: pd.Series, *, reverse: bool = False) -> pd.Series:
        fill = float(series.median()) if series.notna().any() else 0.0
        ranked = series.fillna(fill).rank(method="average", pct=True)
        return 1.0 - ranked if reverse else ranked

    score = (
        FALLBACK_SCORE_WEIGHTS["relative_price_rank"] * rank(relative)
        + FALLBACK_SCORE_WEIGHTS["area_rank"] * rank(area)
        + FALLBACK_SCORE_WEIGHTS["amenity_score"] * amenities.fillna(float(amenities.mean()) if amenities.notna().any() else 0.5)
        + FALLBACK_SCORE_WEIGHTS["newness_score"] * rank(age, reverse=True)
        + FALLBACK_SCORE_WEIGHTS["rooms_rank"] * rank(rooms)
    )
    return work, score


def _three_apartment_tier_labels(
    score: pd.Series,
    economic_quantile: float = FALLBACK_ECONOMIC_QUANTILE,
    luxury_quantile: float = FALLBACK_LUXURY_QUANTILE,
) -> pd.Series:
    if not (0 < economic_quantile < luxury_quantile < 1):
        raise ValueError("Expected 0 < economic_quantile < luxury_quantile < 1.")
    ranked = score.rank(method="first", pct=True)
    bins = [-np.inf, economic_quantile, luxury_quantile, np.inf]
    labels = ["economic", "mid_market", "luxury"]
    return pd.cut(ranked, bins=bins, labels=labels, include_lowest=True).astype("string")


def _apartment_tier_name_map(work: pd.DataFrame, tier: pd.Series) -> dict[str, str]:
    area_all = _optional_numeric_series(work, "primary_area_sqm")
    age_all = _optional_numeric_series(work, "building_age_years")
    rooms_all = _optional_numeric_series(work, "rooms_count_num")
    amenities_all = _amenity_score_series(work)
    refs = {
        "area_q40": float(area_all.quantile(0.40)) if area_all.notna().any() else np.nan,
        "area_q70": float(area_all.quantile(0.70)) if area_all.notna().any() else np.nan,
        "age_q35": float(age_all.quantile(0.35)) if age_all.notna().any() else np.nan,
        "amenity_q65": float(amenities_all.quantile(0.65)) if amenities_all.notna().any() else np.nan,
        "rooms_median": float(rooms_all.median()) if rooms_all.notna().any() else np.nan,
    }
    stats = {key: _semantic_name_components(work.loc[tier == key]) for key in ["economic", "mid_market", "luxury"]}

    econ = stats["economic"]
    if np.isfinite(econ["median_area"]) and np.isfinite(refs["area_q40"]) and econ["median_area"] <= refs["area_q40"]:
        economic_archetype = "\u0648\u0627\u062d\u062f \u06a9\u0648\u0686\u06a9 \u0634\u0647\u0631\u06cc"
    else:
        economic_archetype = "\u062e\u0627\u0646\u0648\u0627\u062f\u06af\u06cc"

    mid = stats["mid_market"]
    if (np.isfinite(mid["median_rooms"]) and mid["median_rooms"] >= 2) or (
        np.isfinite(mid["median_area"]) and np.isfinite(refs["area_q40"]) and mid["median_area"] >= refs["area_q40"]
    ):
        mid_archetype = "\u062e\u0627\u0646\u0648\u0627\u062f\u06af\u06cc"
    else:
        mid_archetype = "\u0648\u0627\u062d\u062f \u06a9\u0648\u0686\u06a9 \u0634\u0647\u0631\u06cc"

    lux = stats["luxury"]
    # Semantic naming does not alter membership. Use stable, domain-readable naming gates
    # rather than a sample-relative quantile so the label does not flip across reruns.
    young = np.isfinite(lux["median_age"]) and lux["median_age"] <= LUXURY_NEWER_MAX_MEDIAN_AGE_YEARS
    amenity_rich = np.isfinite(lux["amenity_score"]) and lux["amenity_score"] >= LUXURY_AMENITY_RICH_MIN_RATE
    family_sized = (np.isfinite(lux["median_rooms"]) and lux["median_rooms"] >= 2) or (
        np.isfinite(lux["median_area"]) and np.isfinite(refs["area_q70"]) and lux["median_area"] >= refs["area_q70"]
    )
    if young and amenity_rich and family_sized:
        luxury_archetype = "\u062e\u0627\u0646\u0648\u0627\u062f\u06af\u06cc \u0646\u0648\u0633\u0627\u0632/\u06a9\u0645\u200c\u0633\u0646 \u0627\u0645\u06a9\u0627\u0646\u0627\u062a\u200c\u062f\u0627\u0631"
    elif family_sized:
        luxury_archetype = "\u062e\u0627\u0646\u0648\u0627\u062f\u06af\u06cc"
    else:
        luxury_archetype = "\u0622\u067e\u0627\u0631\u062a\u0645\u0627\u0646 \u0645\u0645\u062a\u0627\u0632"

    return {
        "economic": f"\u0627\u0642\u062a\u0635\u0627\u062f\u06cc \u0646\u0633\u0628\u06cc | {economic_archetype}",
        "mid_market": f"\u0645\u06cc\u0627\u0646\u200c\u0631\u062f\u0647 | {mid_archetype}",
        "luxury": f"\u0644\u0648\u06a9\u0633 \u0646\u0633\u0628\u06cc | {luxury_archetype}",
    }


def fallback_typology(apartment: pd.DataFrame, min_n: int) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    work, score = _fallback_score(apartment, min_n)
    tier = _three_apartment_tier_labels(score)
    name_map = _apartment_tier_name_map(work, tier)
    segment_id = tier.map({"economic": "SEG01", "mid_market": "SEG02", "luxury": "SEG03"})
    names = tier.map(name_map)
    shares = segment_id.value_counts(normalize=True)
    diag = pd.DataFrame([{
        "specification": "rule_based_fallback",
        "k": int(segment_id.nunique()),
        "silhouette": np.nan,
        "calinski_harabasz": np.nan,
        "davies_bouldin": np.nan,
        "minimum_cluster_share": float(shares.min()) if not shares.empty else np.nan,
        "preliminary_gate_pass": False,
        "stability_ari": np.nan,
        "final_gate_pass": False,
        "notes": "Three apartment business tiers from a fixed composite-score rule; SEG04/SEG05 are compatible land/commercial domain types added outside apartment geometry.",
    }])
    return segment_id, names, diag


def fallback_sensitivity(apartment: pd.DataFrame, min_n: int) -> pd.DataFrame:
    _, score = _fallback_score(apartment, min_n)

    def labels(cuts: tuple[float, float]) -> np.ndarray:
        ranked = score.rank(method="first", pct=True).to_numpy(dtype=float)
        return np.digitize(ranked, bins=np.asarray(cuts, dtype=float), right=True)

    scenarios = [
        ("base_30_80", (0.30, 0.80)),
        ("wider_middle_25_80", (0.25, 0.80)),
        ("more_selective_luxury_30_85", (0.30, 0.85)),
        ("balanced_30_75", (0.30, 0.75)),
    ]
    base = labels(scenarios[0][1])
    rows = []
    for name, cuts in scenarios:
        scenario = labels(cuts)
        shares = pd.Series(scenario).value_counts(normalize=True)
        rows.append({
            "scenario": name,
            "economic_quantile": cuts[0],
            "luxury_quantile": cuts[1],
            "assignment_ari_vs_base": float(adjusted_rand_score(base, scenario)),
            "minimum_type_share": float(shares.min()),
            "notes": "Sensitivity for the three apartment business tiers; land/commercial compatible domain types are unchanged.",
        })
    return pd.DataFrame(rows)


def _assign_compatible_domain_segments(assignments: pd.DataFrame) -> pd.DataFrame:
    out = assignments.copy()
    family = out["property_family"].astype("string").str.lower()
    land_mask = family.isin(["land", "plot", "agriculture", "farm", "garden"])
    commercial_mask = family.isin(["commercial", "shop", "office", "business", "industrial"])

    out.loc[land_mask, "segment_id"] = DOMAIN_SEGMENTS["land"][0]
    out.loc[land_mask, "segment_name"] = DOMAIN_SEGMENTS["land"][1]
    out.loc[land_mask, "segment_method"] = "compatible_domain_segment"

    out.loc[commercial_mask, "segment_id"] = DOMAIN_SEGMENTS["commercial"][0]
    out.loc[commercial_mask, "segment_name"] = DOMAIN_SEGMENTS["commercial"][1]
    out.loc[commercial_mask, "segment_method"] = "compatible_domain_segment"

    other_mask = family.ne("apartment") & ~land_mask & ~commercial_mask
    safe_family = family.fillna("other").astype(str).str.replace(r"[^a-z0-9]+", "_", regex=True).str.strip("_")
    out.loc[other_mask, "segment_id"] = "DOMAIN_" + safe_family.loc[other_mask].str.upper().replace("", "OTHER")
    out.loc[other_mask, "segment_name"] = "Domain: " + family.loc[other_mask].fillna("other").astype(str)
    out.loc[other_mask, "segment_method"] = "compatible_domain_segment"
    return out


def temporal_robustness(apartment: pd.DataFrame, features: list[str], k: int, params: dict[str, Any], seed: int) -> pd.DataFrame:
    month = apartment["analysis_month"].astype(str).str[:7]
    train = apartment.loc[month <= "2024-10"].copy()
    future = apartment.loc[month > "2024-10"].copy()
    if len(train) < k * 50 or future.empty:
        return pd.DataFrame([{"metric": "max_cluster_share_drift_pp", "value": np.nan, "status": "REVIEW", "notes": "Insufficient temporal holdout."}])
    x_train, _, state = fit_transform_state(train, features, params["reference_min_n"], params["min_optional_coverage"])
    model = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=4096, n_init=20).fit(x_train)
    train_labels = model.predict(x_train)
    x_future, _ = transform(future, state)
    future_labels = model.predict(x_future)
    train_share = pd.Series(train_labels).value_counts(normalize=True).reindex(range(k), fill_value=0.0)
    future_share = pd.Series(future_labels).value_counts(normalize=True).reindex(range(k), fill_value=0.0)
    max_drift = float((future_share - train_share).abs().max() * 100.0)
    return pd.DataFrame([{"metric": "max_cluster_share_drift_pp", "value": max_drift, "status": "PASS" if max_drift <= 15 else "REVIEW", "notes": "Fit May-Oct; compare Nov-Dec assignment shares."}])


def profile_segments(assignments: pd.DataFrame, features: pd.DataFrame, text_path: Path) -> pd.DataFrame:
    work = assignments.merge(features, on="source_row_id", how="left", suffixes=("", "_feature"))
    text_flags: list[str] = []
    if text_path.exists():
        text_schema = pl.scan_parquet(text_path).collect_schema().names()
        if "source_row_id" not in text_schema:
            raise ValueError("Text features missing required column: source_row_id")
        text_flags = [
            column
            for column in text_schema
            if column.startswith("keyword_") and column.endswith("_flag")
        ]
        text = pd.read_parquet(
            text_path,
            columns=["source_row_id", *text_flags],
        )
        work = work.merge(text, on="source_row_id", how="left")
    rows: list[dict[str, Any]] = []
    total = len(work)
    numeric = ["sale_price_per_sqm_final_toman", "primary_area_sqm", "rooms_count_num", "building_age_years", "floor_num"]
    amenities = ["has_elevator_bool", "has_parking_bool", "has_warehouse_bool", "has_balcony_bool"]
    for (segment_id, segment_name, method), group in work.groupby(["segment_id", "segment_name", "segment_method"], dropna=False):
        row: dict[str, Any] = {
            "segment_id": segment_id, "segment_name": segment_name, "segment_method": method,
            "listing_n": len(group), "listing_share_pct": len(group) / total * 100.0 if total else np.nan,
            "dominant_property_family": group["property_family"].mode().iloc[0] if "property_family" in group and not group["property_family"].mode().empty else None,
        }
        if "cat3_slug" in group and group["cat3_slug"].notna().any():
            dominant_type = group["cat3_slug"].mode().iloc[0]
            row["dominant_property_type"] = dominant_type
            row["dominant_property_type_share_pct"] = float(group["cat3_slug"].eq(dominant_type).mean() * 100.0)
        else:
            row["dominant_property_type"] = None
            row["dominant_property_type_share_pct"] = np.nan
        if "city_slug" in group.columns:
            city_counts = group["city_slug"].astype("string").dropna().value_counts(normalize=True)
            row["representative_cities"] = "; ".join(f"{city} ({share*100.0:.1f}%)" for city, share in city_counts.head(2).items())
        else:
            row["representative_cities"] = ""
        if {"city_slug", "neighborhood_slug"}.issubset(group.columns):
            location = group["city_slug"].astype("string").fillna("unknown") + " / " + group["neighborhood_slug"].astype("string").fillna("unknown")
            if location.notna().any():
                dominant_location = location.mode().iloc[0]
                row["dominant_location"] = dominant_location
                row["dominant_location_share_pct"] = float(location.eq(dominant_location).mean() * 100.0)
            else:
                row["dominant_location"] = None
                row["dominant_location_share_pct"] = np.nan
        for col in numeric:
            row[f"median_{col}"] = float(pd.to_numeric(group[col], errors="coerce").median()) if col in group and pd.to_numeric(group[col], errors="coerce").notna().any() else np.nan
        amenity_rates: list[float] = []
        for col in amenities + text_flags:
            if col in group:
                values = group[col].map({True: 1.0, False: 0.0, 1: 1.0, 0: 0.0})
                rate = float(values.mean()) if values.notna().any() else np.nan
                row[f"rate_{col}"] = rate
                if col in amenities and np.isfinite(rate):
                    amenity_rates.append(rate)
        row["amenity_index_pct"] = float(np.mean(amenity_rates) * 100.0) if amenity_rates else np.nan
        family_value = str(row.get("dominant_property_family") or "").lower()
        if family_value == "land":
            row["median_rooms_count_num"] = np.nan
            row["median_building_age_years"] = np.nan
            row["median_floor_num"] = np.nan
            row["amenity_index_pct"] = np.nan
        elif family_value == "commercial":
            row["median_rooms_count_num"] = np.nan
            row["amenity_index_pct"] = np.nan
        row["applicability_note"] = "Apartment structural features apply to apartment types; non-applicable land/commercial fields are reported as N/A."
        rows.append(row)
    return pd.DataFrame(rows).sort_values("listing_n", ascending=False)


def run(feature_path: Path = FEATURE_TABLE, text_path: Path = TEXT_TABLE) -> dict[str, Path]:
    feature_path = feature_path.resolve()
    text_path = text_path.resolve()
    if not feature_path.exists():
        raise FileNotFoundError(f"Analysis-ready features not found: {feature_path}")
    seed = int(setting("project", "random_seed", default=42))
    cfg = setting("milestone_3", "segmentation", default={}) or {}
    params = {
        "k_min": max(3, int(cfg.get("k_min", 3))), "k_max": min(PRESENTATION_MAX_CLUSTERS, int(cfg.get("k_max", PRESENTATION_MAX_CLUSTERS))),
        "min_cluster_share": float(cfg.get("min_cluster_share", 0.03)),
        "min_stability_ari": float(cfg.get("min_stability_ari", 0.70)),
        "min_silhouette": float(cfg.get("min_silhouette", 0.15)),
        "min_calinski_harabasz": float(cfg.get("min_calinski_harabasz", 100.0)),
        "max_davies_bouldin": float(cfg.get("max_davies_bouldin", 2.5)),
        "selection_sample": int(cfg.get("selection_sample", 60000)),
        "metric_sample": int(cfg.get("metric_sample", 8000)),
        "stability_train_sample": int(cfg.get("stability_train_sample", 30000)),
        "stability_eval_sample": int(cfg.get("stability_eval_sample", 8000)),
        "stability_runs": int(cfg.get("stability_runs", 3)),
        "reference_min_n": int(cfg.get("reference_min_n", 30)),
        "min_optional_coverage": float(cfg.get("min_optional_feature_coverage", 0.50)),
    }

    if params["k_min"] > params["k_max"]:
        raise ValueError(f"Segmentation k range is invalid after the interpretability cap: {params['k_min']}..{params['k_max']}")

    table_dir = OUTPUTS_DIR / "tables" / "milestone_3" / "market_segmentation"
    qa_dir = OUTPUTS_DIR / "qa" / "milestone_3" / "market_segmentation"
    model_dir = OUTPUTS_DIR / "model_artifacts" / "milestone_3" / "market_segmentation"
    fig_dir = OUTPUTS_DIR / "figures" / "milestone_3" / "market_segmentation"
    for d in [table_dir, qa_dir, model_dir, fig_dir]:
        d.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    show_progress(0, "loading segmentation population")

    required = {
        "source_row_id",
        "analysis_month",
        TARGET,
        "city_slug",
        "neighborhood_slug",
        "cat3_slug",
        "property_family",
        "primary_area_sqm",
        "rooms_count_num",
        "building_age_years",
    }
    optional_source_columns = {
        "construction_year_before_1370_flag",
        *OPTIONAL_FEATURES,
    }
    feature_schema = pl.scan_parquet(feature_path).collect_schema().names()
    missing = sorted(required - set(feature_schema))
    if missing:
        raise ValueError(f"Analysis-ready features missing segmentation columns: {missing}")
    load_columns = [
        column
        for column in feature_schema
        if column in required or column in optional_source_columns
    ]
    frame = pd.read_parquet(
        feature_path,
        columns=load_columns,
    )
    apartment = frame.loc[frame["property_family"] == "apartment"].copy()
    if len(apartment) < params["k_max"] * 100:
        raise RuntimeError(f"Apartment-sale population too small for stable clustering: {len(apartment)}")
    show_progress(12, f"apartment-sale population: {len(apartment):,}")

    diagnostics_parts: list[pd.DataFrame] = []
    specification_results: dict[str, tuple[list[str], int | None]] = {}
    for spec_name, spec_features in [("core", CORE_FEATURES), ("extended", CORE_FEATURES + OPTIONAL_FEATURES)]:
        diagnostics, result = evaluate_specification(apartment, spec_name, spec_features, params, seed)
        diagnostics_parts.append(diagnostics)
        specification_results[spec_name] = (spec_features, result)
        show_progress(35 if spec_name == "core" else 52, f"{spec_name} k-selection complete")
    diagnostics = pd.concat(diagnostics_parts, ignore_index=True)
    selected: tuple[str, list[str], int] | None = None
    for spec_name in ["core", "extended"]:
        spec_features, result = specification_results[spec_name]
        if result is not None:
            selected = (spec_name, spec_features, int(result))
            break

    assignments = frame[["source_row_id", "property_family", "analysis_month", "city_slug", "neighborhood_slug", "cat3_slug"]].copy()
    assignments["segment_id"] = pd.NA
    assignments["segment_name"] = pd.NA
    assignments["segment_method"] = pd.NA
    assignments["clustering_applicable"] = assignments["property_family"].eq("apartment")
    temporal = pd.DataFrame()

    if selected is not None:
        spec_name, requested_features, k = selected
        x_apartment, apartment_transformed, state = fit_transform_state(
            apartment, requested_features, params["reference_min_n"], params["min_optional_coverage"]
        )
        model = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=4096, n_init=20)
        labels = model.fit_predict(x_apartment)
        names = _segment_names(apartment_transformed, labels)
        apartment_assign = pd.DataFrame({
            "source_row_id": apartment["source_row_id"].to_numpy(),
            "segment_id": [f"cluster_{int(v)}" for v in labels],
            "segment_name": [names[int(v)] for v in labels],
            "segment_method": "minibatch_kmeans",
        })
        assignments = assignments.drop(columns=["segment_id", "segment_name", "segment_method"]).merge(apartment_assign, on="source_row_id", how="left")
        assignments = _assign_compatible_domain_segments(assignments)
        method = "minibatch_kmeans"
        temporal = temporal_robustness(apartment, requested_features, k, params, seed)
        selected_k = k
        selected_spec = spec_name
        selected_ari = float(diagnostics.loc[(diagnostics["specification"] == spec_name) & (diagnostics["k"] == k), "stability_ari"].iloc[0])
    else:
        seg_id, seg_name, fallback_diag = fallback_typology(apartment, params["reference_min_n"])
        diagnostics = pd.concat([diagnostics, fallback_diag], ignore_index=True)
        apartment_assign = pd.DataFrame({"source_row_id": apartment["source_row_id"].to_numpy(), "segment_id": seg_id.to_numpy(), "segment_name": seg_name.to_numpy(), "segment_method": "rule_based_descriptive_typology"})
        assignments = assignments.drop(columns=["segment_id", "segment_name", "segment_method"]).merge(apartment_assign, on="source_row_id", how="left")
        assignments = _assign_compatible_domain_segments(assignments)
        method = "rule_based_descriptive_typology"
        primary_ids = assignments["segment_id"].astype(str).isin(PRIMARY_PRESENTATION_IDS)
        selected_k = int(assignments.loc[primary_ids, "segment_id"].nunique())
        selected_spec = "fallback"
        selected_ari = np.nan
        temporal = pd.DataFrame([{"metric": "max_cluster_share_drift_pp", "value": np.nan, "status": "NOT_APPLICABLE", "notes": "Temporal cluster robustness is not claimed for descriptive fallback typology."}])
    show_progress(70, f"primary method: {method}")

    rule_sensitivity = fallback_sensitivity(apartment, params["reference_min_n"])
    profile = profile_segments(assignments, frame, text_path)
    monthly = assignments.groupby(["analysis_month", "segment_id", "segment_name", "segment_method"], dropna=False).size().reset_index(name="listing_n")
    monthly["month_total_n"] = monthly.groupby("analysis_month")["listing_n"].transform("sum")
    monthly["listing_share_pct"] = monthly["listing_n"] / monthly["month_total_n"] * 100.0
    show_progress(82, f"segment profile: {len(profile):,} segments")

    checks: list[Check] = [
        make_check("listing_assignment_complete", "segmentation", int(assignments["segment_id"].notna().sum()), len(assignments), assignments["segment_id"].notna().all()),
        make_check("apartment_population_nonempty", "segmentation", len(apartment), ">0", len(apartment) > 0),
        make_check(
            "stable_clustering_selected", "segmentation", method, "minibatch_kmeans or documented fallback", True,
            critical=False,
            notes="Gates are never lowered. Rule-based output is explicitly descriptive if no stable k passes.",
        ),
        make_check("text_excluded_from_primary_clustering", "method", "post-hoc profile only", "excluded", True),
    ]
    if method == "rule_based_descriptive_typology":
        family_values = set(frame["property_family"].astype("string").str.lower().dropna())
        expected_ids = {"SEG01", "SEG02", "SEG03"}
        if family_values.intersection({"land", "plot", "agriculture", "farm", "garden"}):
            expected_ids.add("SEG04")
        if family_values.intersection({"commercial", "shop", "office", "business", "industrial"}):
            expected_ids.add("SEG05")
        observed_ids = set(assignments.loc[assignments["segment_id"].astype(str).isin(PRIMARY_PRESENTATION_IDS), "segment_id"].astype(str))
        checks.append(make_check(
            "business_typology_ids_stable", "interpretability",
            ",".join(sorted(observed_ids)), ",".join(sorted(expected_ids)), observed_ids == expected_ids,
            notes="SEG01-SEG03 are apartment tiers; SEG04 land/investment and SEG05 commercial/administrative are added only when those compatible families are present.",
        ))

    listing_path = model_dir / "listing_segments.parquet"
    profile_path = table_dir / "segment_profile.csv"
    monthly_path = qa_dir / "segment_monthly_distribution.csv"
    diagnostics_path = qa_dir / "cluster_selection_diagnostics.csv"
    temporal_path = qa_dir / "segment_temporal_robustness.csv"
    rule_sensitivity_path = qa_dir / "rule_based_benchmark_sensitivity.csv"
    checks_path = qa_dir / "segmentation_checks.csv"
    manifest_path = qa_dir / "segmentation_manifest.json"
    atomic_sink_parquet(pl.from_pandas(assignments).lazy(), listing_path)
    atomic_write_csv(pl.from_pandas(profile), profile_path)
    atomic_write_csv(pl.from_pandas(monthly), monthly_path)
    atomic_write_csv(pl.from_pandas(diagnostics), diagnostics_path)
    atomic_write_csv(pl.from_pandas(temporal), temporal_path)
    atomic_write_csv(pl.from_pandas(rule_sensitivity), rule_sensitivity_path)
    atomic_write_csv(checks_frame(checks), checks_path)

    # Keep cluster-selection evidence internal/QA. Professor-facing segmentation figures are limited to two clear views.
    qa_fig_dir = qa_dir / "figures"
    qa_fig_dir.mkdir(parents=True, exist_ok=True)
    stale_diagnostic = fig_dir / "cluster_selection_quality_stability.png"
    stale_heatmap = fig_dir / "standardized_segment_profile_heatmap.png"
    for stale in [stale_diagnostic, stale_heatmap]:
        if stale.exists():
            stale.unlink()

    diagnostic_fig = qa_fig_dir / "cluster_selection_quality_stability.png"
    plotted = diagnostics.loc[diagnostics["specification"].isin(["core", "extended"])].copy()
    if not plotted.empty:
        fig, ax = plt.subplots(figsize=(11.2, 6.2))
        spec_labels = {"core": "Core features", "extended": "Extended features"}
        spec_colors = {"core": "#2F6B9A", "extended": "#C96B2C"}
        for spec, part in plotted.groupby("specification"):
            label = spec_labels.get(spec, str(spec).title())
            color = spec_colors.get(spec, "#666666")
            ax.plot(part["k"], part["silhouette"], marker="o", linewidth=2.2, color=color, label=f"{label} — silhouette")
            valid_ari = part["stability_ari"].notna()
            if valid_ari.any():
                ax.plot(part.loc[valid_ari, "k"], part.loc[valid_ari, "stability_ari"], marker="s", linewidth=1.8, linestyle="--", color=color, alpha=0.8, label=f"{label} — stability ARI")
        ax.axhline(params["min_silhouette"], linewidth=1.1, linestyle="--", color="#777777", label=f"Silhouette gate ({params['min_silhouette']:.2f})")
        ax.axhline(params["min_stability_ari"], linewidth=1.1, linestyle=":", color="#444444", label=f"Stability gate ({params['min_stability_ari']:.2f})")
        ax.set_xlabel("Candidate cluster count (k)")
        ax.set_ylabel("Quality / stability score")
        ax.grid(axis="y", alpha=0.18, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title("Apartment-Sale Cluster Selection Diagnostic", loc="left", fontsize=15, fontweight="bold", pad=12)
        method_text = "Descriptive typology fallback used — no candidate passed all predeclared gates" if method == "rule_based_descriptive_typology" else f"Selected clustering method: {method.replace('_', ' ')}"
        ax.text(0, 1.01, method_text, transform=ax.transAxes, fontsize=9.5, va="bottom", color="#555555")
        ax.legend(frameon=False, ncol=2, fontsize=8.5, loc="best")
        fig.text(0.01, 0.012, "Internal QA evidence. Gates are predeclared and are not relaxed when no candidate passes.", fontsize=8.5, color="#555555")
        fig.tight_layout(rect=(0, 0.045, 1, 0.98))
        fig.savefig(diagnostic_fig, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    display = _presentation_segments(profile, maximum=5)

    stale_figures = [
        fig_dir / "segment_positioning_listing_count_vs_median_psm.png",
        fig_dir / "segment_typology_profile_cards.png",
        fig_dir / "segment_profile_clustered_bar.png",
    ]
    for stale in stale_figures:
        if stale.exists():
            stale.unlink()

    positioning_fig = fig_dir / "segment_positioning_by_area_and_normalized_price.png"
    profile_bar_fig = fig_dir / "segment_profile_summary_by_relative_price.png"

    presentation_base = apply_compatible_reference(frame, params["reference_min_n"])
    presentation_base = presentation_base.merge(
        assignments[["source_row_id", "segment_id", "segment_name", "segment_method"]],
        on="source_row_id", how="left",
    )
    keep_ids = set(display["segment_id"].astype(str)) if not display.empty else set()
    presentation_base = presentation_base.loc[presentation_base["segment_id"].astype(str).isin(keep_ids)].copy()
    position = presentation_base.groupby("segment_id", dropna=False).agg(
        listing_n=("source_row_id", "size"),
        median_area_sqm=("primary_area_sqm", "median"),
        median_normalized_price_index=("normalized_price_index", "median"),
    ).reset_index()
    position = position.merge(
        display[["segment_id", "segment_label_fa", "segment_name"]], on="segment_id", how="left"
    ) if not display.empty else position

    if not position.empty:
        position = position.dropna(subset=["median_area_sqm", "median_normalized_price_index"]).copy()
        counts = position["listing_n"].to_numpy(dtype=float)
        size_base = counts / max(float(np.nanmax(counts)), 1.0)
        bubble_sizes = 520 + 1450 * np.sqrt(size_base)
        segment_colors = {
            "SEG01": "#5D7FA3",
            "SEG02": "#3F6F99",
            "SEG03": "#7595B3",
            "SEG04": "#9CA9B5",
            "SEG05": "#C8793D",
        }
        # Fixed annotation directions keep labels away from the title and from one another.
        offsets = {
            "SEG01": (24, 8),
            "SEG02": (20, 20),
            "SEG03": (22, -48),
            "SEG04": (20, -34),
            "SEG05": (12, 22),
        }
        fig, ax = plt.subplots(figsize=(13.2, 8.6))
        fig.suptitle(
            "Segment positioning by area and normalized asking price",
            fontsize=15.8, fontweight="bold", y=0.965,
        )
        fig.text(0.5, 0.925, "Bubble size reflects listing count", ha="center", va="center", fontsize=11.0, color="#44515E")
        for (_, row), size in zip(position.iterrows(), bubble_sizes):
            sid = str(row["segment_id"])
            color = segment_colors.get(sid, "#5D7FA3")
            x_value = float(row["median_area_sqm"])
            y_value = float(row["median_normalized_price_index"])
            ax.scatter(
                x_value, y_value, s=float(size), color=color, alpha=0.88,
                edgecolors="white", linewidths=1.2, zorder=3,
            )
            label = (
                f"{sid}\n{row['segment_label_fa']}\n"
                f"Area={x_value:.0f} m\u00b2 | Price idx={y_value:.1f} | N={int(row['listing_n']):,}"
            )
            xytext = offsets.get(sid, (14, 12))
            ax.annotate(
                label, (x_value, y_value), xytext=xytext, textcoords="offset points",
                fontsize=8.9, annotation_clip=False, zorder=4,
                bbox={"boxstyle": "round,pad=0.34", "facecolor": "white", "edgecolor": "#B8C5D0", "alpha": 0.97},
                arrowprops={"arrowstyle": "-", "color": color, "linewidth": 1.0},
            )
        ax.axhline(100.0, color="#66788A", linewidth=1.15, linestyle="--")
        ax.set_xlabel("Median area (sqm)")
        ax.set_ylabel("Median normalized price index\n(100 = local reference within comparable property group)")
        ax.grid(alpha=0.16, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        if len(position):
            y_values = pd.to_numeric(position["median_normalized_price_index"], errors="coerce").dropna().to_numpy(dtype=float)
            if y_values.size:
                y_min, y_max = float(np.min(y_values)), float(np.max(y_values))
                y_span = max(y_max - y_min, 20.0)
                ax.set_ylim(y_min - 0.12 * y_span, y_max + 0.10 * y_span)
        fig.text(
            0.01, 0.018,
            "SEG01-SEG03 are apartment descriptive tiers when clustering gates fail; SEG04-SEG05 are compatible land/commercial domain types. These are market types, not statistical clusters.",
            fontsize=8.3, color="#555555",
        )
        fig.subplots_adjust(left=0.095, right=0.965, bottom=0.11, top=0.84)
        fig.savefig(positioning_fig, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    required_profile = {
        "median_primary_area_sqm", "median_rooms_count_num", "median_building_age_years",
        "amenity_index_pct", "dominant_property_type", "representative_cities",
    }
    if not display.empty and required_profile.issubset(display.columns):
        profile_view = display.merge(
            position[["segment_id", "median_normalized_price_index"]], on="segment_id", how="left"
        )
        profile_view = profile_view.sort_values("median_normalized_price_index", ascending=False, na_position="last").reset_index(drop=True)
        n_rows = len(profile_view)
        y = np.arange(n_rows)
        colors = [
            {"SEG01": "#5D7FA3", "SEG02": "#3F6F99", "SEG03": "#7595B3", "SEG04": "#9CA9B5", "SEG05": "#C8793D"}.get(str(sid), "#5D7FA3")
            for sid in profile_view["segment_id"]
        ]
        fig, ax = plt.subplots(figsize=(13.4, max(6.6, 1.35 * n_rows + 1.4)))
        values = pd.to_numeric(profile_view["median_normalized_price_index"], errors="coerce").to_numpy(dtype=float)
        bars = ax.barh(y, values, color=colors, height=0.62)
        ax.axvline(100.0, color="#66788A", linewidth=1.05, linestyle="--")
        ax.set_yticks(y, profile_view["segment_id"].astype(str).tolist(), fontsize=10.5)
        ax.invert_yaxis()
        ax.set_xlabel("Median normalized price index")
        ax.grid(axis="x", alpha=0.14, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title("Segment profile summary ordered by relative price level", fontsize=15.0, fontweight="bold", pad=14)
        max_value = max(float(np.nanmax(values)) if np.isfinite(values).any() else 100.0, 100.0)
        ax.set_xlim(0, max_value * 1.52)

        def _fmt_numeric(value: object, digits: int = 0, suffix: str = "") -> str:
            number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
            return f"{float(number):,.{digits}f}{suffix}" if pd.notna(number) else "N/A"

        for bar, (_, row), value in zip(bars, profile_view.iterrows(), values):
            family = str(row.get("dominant_property_family") or "").lower()
            area = _fmt_numeric(row.get("median_primary_area_sqm"), 0, " m\u00b2")
            rooms = _fmt_numeric(row.get("median_rooms_count_num"), 1) if family == "apartment" else "N/A"
            age = _fmt_numeric(row.get("median_building_age_years"), 1, " yrs") if family != "land" else "N/A"
            amenity = _fmt_numeric(row.get("amenity_index_pct"), 1, "/100") if family == "apartment" else "N/A"
            ptype = str(row.get("dominant_property_type") or "N/A")
            cities = str(row.get("representative_cities") or "N/A")
            ax.text(
                max(value - max_value * 0.025, 1.0), bar.get_y() + bar.get_height() / 2,
                f"{value:.1f}", va="center", ha="right", fontsize=9.3, fontweight="bold", color="white",
            )
            descriptor = (
                f"{row['segment_label_fa']}\n"
                f"Type: {ptype} | Area: {area} | Rooms: {rooms} | Amenities: {amenity}\n"
                f"Age: {age} | Cities: {cities} | N={int(row['listing_n']):,}"
            )
            ax.text(
                value + max_value * 0.018, bar.get_y() + bar.get_height() / 2, descriptor,
                va="center", ha="left", fontsize=8.9, color="#273746",
            )
        fig.text(
            0.01, 0.012,
            "Normalized price uses neighborhood x property-family reference when reliable, then city x property-family and family-level fallbacks. Non-applicable structural fields are shown as N/A.",
            fontsize=8.3, color="#555555",
        )
        fig.tight_layout(rect=(0, 0.045, 1, 0.98))
        fig.savefig(profile_bar_fig, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    status = summarize_checks(checks)
    atomic_write_json(
        {
            "version": VERSION,
            "status": status,
            "input": relative_to_project(feature_path),
            "primary_method": method,
            "selected_specification": selected_spec,
            "selected_k": selected_k,
            "selected_stability_ari": selected_ari if np.isfinite(selected_ari) else None,
            "interpretability_cap_k": PRESENTATION_MAX_CLUSTERS,
            "semantic_naming": "stable business IDs: SEG01 economic/small urban apartment, SEG02 mid-market/family apartment, SEG03 relative-luxury apartment; SEG03 receives the newer/amenity-rich family label when median age <= 5 years, amenity rate >= 80%, and family-sized profile applies; SEG04 land/investment and SEG05 commercial/administrative compatible domain types",
            "presentation_interpretation": "When primary_method is rule_based_descriptive_typology, SEG01-SEG03 are fixed-rule apartment tiers and SEG04-SEG05 are compatible domain types. selected_k counts surfaced market types, not statistical clusters.",
            "gates": {key: params[key] for key in ["min_cluster_share", "min_stability_ari", "min_silhouette", "min_calinski_harabasz", "max_davies_bouldin"]},
            "outputs": {"listing_segments": relative_to_project(listing_path), "segment_profile": relative_to_project(profile_path), "diagnostics": relative_to_project(diagnostics_path), "temporal_robustness": relative_to_project(temporal_path), "rule_based_benchmark_sensitivity": relative_to_project(rule_sensitivity_path), "checks": relative_to_project(checks_path), "presentation_scatter": relative_to_project(positioning_fig) if positioning_fig.exists() else None, "presentation_profile": relative_to_project(profile_bar_fig) if profile_bar_fig.exists() else None, "presentation_profile_bar_legacy_alias": relative_to_project(profile_bar_fig) if profile_bar_fig.exists() else None, "qa_diagnostic_figure": relative_to_project(diagnostic_fig) if diagnostic_fig.exists() else None},
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }, manifest_path,
    )
    show_progress(100, f"complete in {time.perf_counter()-started:.1f}s", final=True)
    return {"listing_segments": listing_path, "profile": profile_path, "monthly": monthly_path, "diagnostics": diagnostics_path, "temporal": temporal_path, "rule_sensitivity": rule_sensitivity_path, "checks": checks_path, "manifest": manifest_path, "positioning_figure": positioning_fig, "profile_figure": profile_bar_fig, "profile_bar_figure": profile_bar_fig, "qa_diagnostic_figure": diagnostic_fig}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run apartment clustering QA with stable business-typology fallback and compatible domain segments.")
    parser.add_argument("--features", type=Path, default=FEATURE_TABLE)
    parser.add_argument("--text-features", type=Path, default=TEXT_TABLE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run(args.features, args.text_features)
    print("M3 MARKET SEGMENTATION COMPLETED")
    for name, path in outputs.items():
        if path.exists():
            print(f"{name}: {relative_to_project(path)}")


if __name__ == "__main__":
    main()
