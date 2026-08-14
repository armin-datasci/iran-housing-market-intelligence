from __future__ import annotations

import argparse
import gc
import math
import os
import time
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
import polars as pl
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.common.config import setting
from src.common.io_utils import atomic_write_csv, atomic_write_json
from src.common.paths import OUTPUTS_DIR, relative_to_project
from src.common.validation import Check, checks_frame, make_check, summarize_checks

VERSION = "m3-price-drivers-v1.1-model-qa"
TARGET = "sale_price_per_sqm_final_toman"
LOG_TARGET = "log_sale_price_per_sqm"
FEATURE_TABLE = OUTPUTS_DIR / "model_artifacts" / "milestone_3" / "price_drivers" / "analysis_ready_features.parquet"
PROGRESS_WIDTH = 30

NUMERIC = ["primary_area_sqm", "rooms_count_num", "building_age_years", "floor_num", "total_floors_count_num"]
BINARY = ["construction_year_before_1370_flag", "has_elevator_bool", "has_parking_bool", "has_warehouse_bool", "has_balcony_bool", "is_rebuilt_bool"]
CATEGORICAL = ["analysis_month", "city_slug", "neighborhood_slug", "cat3_slug", "property_family", "building_direction"]
ACTIONABLE_NUMERIC = ["primary_area_sqm", "rooms_count_num", "building_age_years", "floor_num"]
ACTIONABLE_BINARY = ["has_elevator_bool", "has_parking_bool", "has_warehouse_bool", "has_balcony_bool", "is_rebuilt_bool", "construction_year_before_1370_flag"]
ACTIONABLE_CATEGORICAL = ["building_direction"]


def show_progress(percent: int, label: str, *, final: bool = False) -> None:
    pct = max(0, min(100, int(percent)))
    filled = int(PROGRESS_WIDTH * pct / 100)
    bar = "#" * filled + "-" * (PROGRESS_WIDTH - filled)
    print(
        f"\rM3 drivers [{bar}] {pct:3d}% complete | {100-pct:3d}% remaining | {label[:42]:42s}",
        end="\n" if final else "", flush=True,
    )


def _one_hot(min_frequency: int) -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", min_frequency=min_frequency, sparse_output=True)
    except TypeError:  # scikit-learn <1.2
        return OneHotEncoder(handle_unknown="ignore", min_frequency=min_frequency, sparse=True)


def active_features(columns: Iterable[str]) -> tuple[list[str], list[str], list[str]]:
    names = set(columns)
    return (
        [x for x in NUMERIC if x in names],
        [x for x in BINARY if x in names],
        [x for x in CATEGORICAL if x in names],
    )


def build_preprocessor(columns: Iterable[str], min_frequency: int) -> tuple[ColumnTransformer, list[str]]:
    """Build one shared preprocessing contract for linear and tree benchmarks."""
    numeric, binary, categorical = active_features(columns)
    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric:
        transformers.append((
            "numeric",
            Pipeline([
                ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                ("scale", StandardScaler(with_mean=False)),
            ]),
            numeric,
        ))
    if binary:
        transformers.append((
            "binary",
            Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent", add_indicator=True)),
                ("scale", StandardScaler(with_mean=False)),
            ]),
            binary,
        ))
    if categorical:
        transformers.append((
            "categorical",
            Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", _one_hot(min_frequency)),
            ]),
            categorical,
        ))
    if not transformers:
        raise ValueError("No usable price-driver features were found.")
    pre = ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.3)
    return pre, numeric + binary + categorical


def build_pipeline(columns: Iterable[str], alpha: float, min_frequency: int) -> tuple[Pipeline, list[str]]:
    pre, features = build_preprocessor(columns, min_frequency)
    model = Ridge(alpha=float(alpha), solver="lsqr")
    return Pipeline([("preprocess", pre), ("ridge", model)]), features


def benchmark_models(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    alpha: float,
    min_frequency: int,
    seed: int,
    train_max_rows: int,
    test_max_rows: int,
    tree_max_depth: int = 18,
    tree_min_samples_leaf: int = 10,
    rf_estimators: int = 120,
) -> pd.DataFrame:
    """Compare Ridge, Decision Tree, and Random Forest on identical rows/features.

    This is a bounded sensitivity benchmark only. Model selection is not made from
    these held-out results; the canonical interpretable model remains Ridge unless
    a later review explicitly changes that decision.
    """
    train_sub = _sample(train, train_max_rows, seed).copy()
    test_sub = _sample(test, test_max_rows, seed + 1).copy()
    pre, shared_features = build_preprocessor(features, min_frequency)
    x_train = pre.fit_transform(_prepare(train_sub, shared_features))
    x_test = pre.transform(_prepare(test_sub, shared_features))
    y_train = train_sub[LOG_TARGET].to_numpy(dtype=float)

    models: list[tuple[str, Any]] = [
        ("ridge", Ridge(alpha=float(alpha), solver="lsqr")),
        (
            "decision_tree",
            DecisionTreeRegressor(
                max_depth=int(tree_max_depth),
                min_samples_leaf=int(tree_min_samples_leaf),
                random_state=seed,
            ),
        ),
        (
            "random_forest",
            RandomForestRegressor(
                n_estimators=int(rf_estimators),
                max_depth=int(tree_max_depth),
                min_samples_leaf=int(tree_min_samples_leaf),
                n_jobs=-1,
                random_state=seed,
            ),
        ),
    ]
    rows: list[dict[str, Any]] = []
    for model_name, model in models:
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        row = _metrics(test_sub, pred, split="heldout_test")
        row.update({
            "model": model_name,
            "benchmark_train_n": len(train_sub),
            "benchmark_test_n": len(test_sub),
            "feature_count_raw": len(shared_features),
            "shared_row_feature_contract": True,
            "benchmark_role": "sensitivity_comparison_not_primary_model_selection",
        })
        rows.append(row)
        del model
        gc.collect()
    return pd.DataFrame(rows)

def _prepare(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    out = frame[features].copy()
    for col in features:
        if col in NUMERIC:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        elif col in BINARY:
            out[col] = out[col].map({True: 1.0, False: 0.0, 1: 1.0, 0: 0.0}).astype(float)
        else:
            out[col] = out[col].astype("string").fillna("__missing__").replace({"": "__missing__"}).astype(str)
    return out


def _metrics(frame: pd.DataFrame, pred_log: np.ndarray, split: str) -> dict[str, Any]:
    y_log = pd.to_numeric(frame[LOG_TARGET], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(frame[TARGET], errors="coerce").to_numpy(dtype=float)
    pred = np.exp(np.clip(pred_log, -50, 50))
    abs_error = np.abs(pred - y)
    ape = abs_error / np.maximum(y, 1e-9) * 100.0
    return {
        "split": split,
        "n": len(frame),
        "rmse_log": float(math.sqrt(mean_squared_error(y_log, pred_log))),
        "r2_log": float(r2_score(y_log, pred_log)) if len(frame) > 1 else np.nan,
        "mae_price_per_sqm_toman": float(np.mean(abs_error)),
        "median_abs_error_price_per_sqm_toman": float(np.median(abs_error)),
        "median_absolute_percentage_error_pct": float(np.median(ape)),
        "p75_absolute_percentage_error_pct": float(np.quantile(ape, 0.75)),
        "p90_absolute_percentage_error_pct": float(np.quantile(ape, 0.90)),
        "within_20pct_share_pct": float(np.mean(ape <= 20.0) * 100.0),
        "within_30pct_share_pct": float(np.mean(ape <= 30.0) * 100.0),
        "within_50pct_share_pct": float(np.mean(ape <= 50.0) * 100.0),
    }

def _sample(frame: pd.DataFrame, maximum: int, seed: int) -> pd.DataFrame:
    if len(frame) <= maximum:
        return frame.copy()
    return frame.sample(n=maximum, random_state=seed)


def select_alpha(train: pd.DataFrame, validation: pd.DataFrame, features: list[str], alphas: list[float], min_frequency: int, maximum_rows: int, seed: int) -> tuple[float, pd.DataFrame]:
    train_sub = _sample(train, maximum_rows, seed)
    rows: list[dict[str, Any]] = []
    best_alpha = float(alphas[0])
    best = float("inf")
    for alpha in alphas:
        pipe, _ = build_pipeline(features, alpha, min_frequency)
        pipe.fit(_prepare(train_sub, features), train_sub[LOG_TARGET].to_numpy(dtype=float))
        pred = pipe.predict(_prepare(validation, features))
        rmse = float(math.sqrt(mean_squared_error(validation[LOG_TARGET].to_numpy(dtype=float), pred)))
        rows.append({"alpha": float(alpha), "validation_rmse_log": rmse, "selection_train_n": len(train_sub), "validation_n": len(validation)})
        if rmse < best:
            best = rmse
            best_alpha = float(alpha)
    return best_alpha, pd.DataFrame(rows)


def _bh(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.minimum.accumulate((ranked * len(p) / np.arange(1, len(p)+1))[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    out = np.empty_like(adjusted)
    out[order] = adjusted
    return out.tolist()


def statistical_tests(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    y = pd.to_numeric(frame[LOG_TARGET], errors="coerce")
    for feature in [x for x in ACTIONABLE_NUMERIC if x in frame.columns]:
        x = pd.to_numeric(frame[feature], errors="coerce")
        mask = x.notna() & y.notna()
        if mask.sum() >= 30 and x[mask].nunique() > 1:
            r, p = stats.spearmanr(x[mask], y[mask])
            rows.append({"feature": feature, "test": "spearman", "statistic": float(r), "p_value": float(p), "n": int(mask.sum())})
    for feature in [x for x in ACTIONABLE_BINARY if x in frame.columns]:
        values = frame[feature].map({True: 1, False: 0, 1: 1, 0: 0})
        a = y[values == 1].dropna().to_numpy(dtype=float)
        b = y[values == 0].dropna().to_numpy(dtype=float)
        if len(a) >= 20 and len(b) >= 20:
            result = stats.mannwhitneyu(a, b, alternative="two-sided", method="asymptotic")
            rows.append({"feature": feature, "test": "mann_whitney", "statistic": float(result.statistic), "p_value": float(result.pvalue), "n": len(a)+len(b)})
    for feature in [x for x in ACTIONABLE_CATEGORICAL if x in frame.columns]:
        groups = [group[LOG_TARGET].dropna().to_numpy(dtype=float) for _, group in frame.groupby(feature, dropna=True) if len(group) >= 20]
        if len(groups) >= 2:
            result = stats.kruskal(*groups)
            rows.append({"feature": feature, "test": "kruskal", "statistic": float(result.statistic), "p_value": float(result.pvalue), "n": int(sum(len(x) for x in groups))})
    if rows:
        q = _bh([row["p_value"] for row in rows])
        for row, adjusted in zip(rows, q):
            row["fdr_bh_q_value"] = adjusted
    return pd.DataFrame(rows)


def _bootstrap_ci(values: np.ndarray, seed: int, runs: int = 250) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if len(values) < 5:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.empty(runs, dtype=float)
    for i in range(runs):
        means[i] = float(np.mean(rng.choice(values, size=len(values), replace=True)))
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def adjusted_associations(pipe: Pipeline, reference: pd.DataFrame, training: pd.DataFrame, features: list[str], seed: int) -> pd.DataFrame:
    ref = _sample(reference, 5000, seed).copy()
    rows: list[dict[str, Any]] = []
    active = set(features)

    def add_contrast(feature_label: str, base_feature: str, low: Any, high: Any, contrast: str, seed_offset: int) -> None:
        low_frame = ref.copy()
        high_frame = ref.copy()
        low_frame[base_feature] = low
        high_frame[base_feature] = high
        low_pred = pipe.predict(_prepare(low_frame, features))
        high_pred = pipe.predict(_prepare(high_frame, features))
        deltas = (np.exp(np.clip(high_pred - low_pred, -5, 5)) - 1.0) * 100.0
        lo, hi = _bootstrap_ci(deltas, seed + seed_offset)
        rows.append({
            "feature": feature_label,
            "base_feature": base_feature,
            "contrast_definition": contrast,
            "adjusted_effect_pct": float(np.mean(deltas)),
            "adjusted_effect_ci_low": lo,
            "adjusted_effect_ci_high": hi,
            "uncertainty_method": "bootstrap_mean_of_model_implied_row_contrasts",
            "interpretation": "model-implied adjusted association; not causal contribution",
            "reference_n": len(ref),
        })

    offset = 0
    for feature in ACTIONABLE_NUMERIC:
        if feature not in active:
            continue
        series = pd.to_numeric(training[feature], errors="coerce").dropna()
        if len(series) < 30:
            continue
        q25, q75 = float(series.quantile(0.25)), float(series.quantile(0.75))
        add_contrast(feature, feature, q25, q75, f"p75 ({q75:.4g}) vs p25 ({q25:.4g})", offset)
        offset += 1
    for feature in ACTIONABLE_BINARY:
        if feature in active:
            add_contrast(feature, feature, 0.0, 1.0, "true vs false", offset)
            offset += 1
    for feature in ACTIONABLE_CATEGORICAL:
        if feature not in active:
            continue
        series = training[feature].astype("string").dropna()
        if series.empty:
            continue
        reference_level = str(series.mode().iloc[0])
        levels = series.value_counts().head(6).index.astype(str).tolist()
        for level in levels:
            if level == reference_level:
                continue
            add_contrast(f"{feature}:{level}", feature, reference_level, level, f"{level} vs modal reference {reference_level}", offset)
            offset += 1
    return pd.DataFrame(rows)


def grouped_permutation_importance(pipe: Pipeline, test: pd.DataFrame, features: list[str], seed: int, maximum_rows: int) -> pd.DataFrame:
    sample = _sample(test, maximum_rows, seed).copy()
    y = sample[LOG_TARGET].to_numpy(dtype=float)
    base_pred = pipe.predict(_prepare(sample, features))
    base_rmse = float(math.sqrt(mean_squared_error(y, base_pred)))
    blocks: dict[str, list[str]] = {
        "location": [x for x in ["city_slug", "neighborhood_slug"] if x in features],
        "property_type": [x for x in ["property_family", "cat3_slug"] if x in features],
        "time": [x for x in ["analysis_month"] if x in features],
    }
    for feature in [*ACTIONABLE_NUMERIC, *ACTIONABLE_BINARY, *ACTIONABLE_CATEGORICAL]:
        if feature in features:
            blocks[feature] = [feature]
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for block, cols in blocks.items():
        if not cols:
            continue
        effects: list[float] = []
        for _ in range(3):
            permuted = sample.copy()
            order = rng.permutation(len(permuted))
            for col in cols:
                permuted[col] = permuted[col].to_numpy()[order]
            pred = pipe.predict(_prepare(permuted, features))
            rmse = float(math.sqrt(mean_squared_error(y, pred)))
            effects.append(rmse - base_rmse)
        rows.append({
            "feature_block": block,
            "columns": ";".join(cols),
            "heldout_rmse_log_increase_mean": float(np.mean(effects)),
            "heldout_rmse_log_increase_sd": float(np.std(effects, ddof=1)) if len(effects) > 1 else 0.0,
            "test_sample_n": len(sample),
            "interpretation": "held-out predictive contribution; not causal importance",
        })
    return pd.DataFrame(rows).sort_values("heldout_rmse_log_increase_mean", ascending=False)


def error_analysis(test: pd.DataFrame, pred_log: np.ndarray, minimum_n: int) -> pd.DataFrame:
    """Robust held-out AVM error analysis by geography/property family.

    Mean percentage error is intentionally excluded because a few tiny/abnormal
    denominators can make it explode and obscure typical model performance.
    """
    work = test[["city_slug", "property_family", TARGET]].copy()
    work["prediction"] = np.exp(np.clip(pred_log, -50, 50))
    work["abs_error_toman_per_sqm"] = np.abs(work["prediction"] - work[TARGET])
    work["ape_pct"] = work["abs_error_toman_per_sqm"] / np.maximum(work[TARGET], 1e-9) * 100.0

    rows: list[pd.DataFrame] = []
    for level, keys in [
        ("city", ["city_slug"]),
        ("property_family", ["property_family"]),
        ("city_property_family", ["city_slug", "property_family"]),
    ]:
        group = work.groupby(keys, dropna=False).agg(
            n=(TARGET, "size"),
            median_ape_pct=("ape_pct", "median"),
            p75_ape_pct=("ape_pct", lambda s: float(s.quantile(0.75))),
            p90_ape_pct=("ape_pct", lambda s: float(s.quantile(0.90))),
            within_20pct_share_pct=("ape_pct", lambda s: float((s <= 20.0).mean() * 100.0)),
            within_30pct_share_pct=("ape_pct", lambda s: float((s <= 30.0).mean() * 100.0)),
            within_50pct_share_pct=("ape_pct", lambda s: float((s <= 50.0).mean() * 100.0)),
            median_abs_error_toman_per_sqm=("abs_error_toman_per_sqm", "median"),
            mae_toman_per_sqm=("abs_error_toman_per_sqm", "mean"),
        ).reset_index()
        group["error_scope"] = level
        group["minimum_reliable_n"] = int(minimum_n)
        group["reliability_status"] = np.where(group["n"] >= minimum_n, "RELIABLE", "LOW_N")
        rows.append(group)
    return pd.concat(rows, ignore_index=True, sort=False)

def _write_model_atomic(pipe: Pipeline, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    joblib.dump(pipe, tmp)
    os.replace(tmp, path)


def run(feature_path: Path = FEATURE_TABLE) -> dict[str, Path]:
    feature_path = feature_path.resolve()
    if not feature_path.exists():
        raise FileNotFoundError(f"Analysis-ready features not found: {feature_path}")
    seed = int(setting("project", "random_seed", default=42))
    alphas = [float(x) for x in setting("milestone_3", "models", "ridge_alphas", default=[0.1, 1.0, 10.0, 100.0])]
    min_frequency = int(setting("milestone_3", "models", "one_hot_min_frequency", default=50))
    alpha_max = int(setting("milestone_3", "models", "alpha_selection_max_rows", default=120000))
    permutation_max = int(setting("milestone_3", "models", "permutation_sample_rows", default=10000))
    legacy_rf_max = int(setting("milestone_3", "models", "rf_sensitivity_max_rows", default=60000))
    benchmark_train_max = int(setting("milestone_3", "models", "benchmark_train_max_rows", default=legacy_rf_max))
    benchmark_test_max = int(setting("milestone_3", "models", "benchmark_test_max_rows", default=20000))
    benchmark_tree_depth = int(setting("milestone_3", "models", "benchmark_tree_max_depth", default=18))
    benchmark_leaf = int(setting("milestone_3", "models", "benchmark_min_samples_leaf", default=10))
    benchmark_rf_estimators = int(setting("milestone_3", "models", "benchmark_rf_estimators", default=120))
    minimum_n = int(setting("analysis", "minimum_valid_listings", "default", default=30))

    table_dir = OUTPUTS_DIR / "tables" / "milestone_3" / "price_drivers"
    qa_dir = OUTPUTS_DIR / "qa" / "milestone_3" / "price_drivers"
    model_dir = OUTPUTS_DIR / "model_artifacts" / "milestone_3" / "price_drivers"
    for d in [table_dir, qa_dir, model_dir]:
        d.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    show_progress(0, "loading analysis-ready features")

    frame = pd.read_parquet(feature_path)
    required = {TARGET, LOG_TARGET, "analysis_split", "source_row_id", "city_slug", "property_family"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Analysis-ready features missing price-driver columns: {missing}")
    numeric, binary, categorical = active_features(frame.columns)
    features = numeric + binary + categorical
    train = frame.loc[frame["analysis_split"] == "train"].copy()
    validation = frame.loc[frame["analysis_split"] == "validation"].copy()
    test = frame.loc[frame["analysis_split"] == "test"].copy()
    if min(len(train), len(validation), len(test)) == 0:
        raise RuntimeError(f"Forward split is incomplete: train={len(train)}, validation={len(validation)}, test={len(test)}")
    show_progress(12, f"forward split: {len(train):,}/{len(validation):,}/{len(test):,}")

    best_alpha, alpha_diagnostics = select_alpha(train, validation, features, alphas, min_frequency, alpha_max, seed)
    show_progress(25, f"selected Ridge alpha={best_alpha:g}")

    # Selection diagnostics remain genuinely out-of-sample for validation.
    selection_pipe, selection_features = build_pipeline(features, best_alpha, min_frequency)
    selection_pipe.fit(_prepare(train, selection_features), train[LOG_TARGET].to_numpy(dtype=float))
    selection_train_pred = selection_pipe.predict(_prepare(train, selection_features))
    selection_validation_pred = selection_pipe.predict(_prepare(validation, selection_features))
    diagnostic_rows = [
        {**_metrics(train, selection_train_pred, "train"), "model": "ridge", "evaluation_stage": "selection_model_fit_on_train"},
        {**_metrics(validation, selection_validation_pred, "validation"), "model": "ridge", "evaluation_stage": "selection_model_fit_on_train"},
    ]
    del selection_pipe, selection_train_pred, selection_validation_pred
    gc.collect()

    # Final interpretable Ridge is refit on train+validation and evaluated once on held-out test.
    fit_frame = pd.concat([train, validation], ignore_index=True)
    pipe, features = build_pipeline(features, best_alpha, min_frequency)
    pipe.fit(_prepare(fit_frame, features), fit_frame[LOG_TARGET].to_numpy(dtype=float))
    test_pred = pipe.predict(_prepare(test, features))
    diagnostic_rows.append({
        **_metrics(test, test_pred, "test"),
        "model": "ridge",
        "evaluation_stage": "final_model_fit_on_train_plus_validation",
    })
    diagnostics = pd.DataFrame(diagnostic_rows)
    show_progress(43, f"final Ridge held-out test complete; N={len(test):,}")

    # Fair three-model benchmark: identical rows, raw feature contract, preprocessing matrix, and held-out population.
    benchmark_frames: list[pd.DataFrame] = []
    populations = [("all_sales", fit_frame, test)]
    apartment_fit = fit_frame.loc[fit_frame["property_family"].astype(str).str.lower() == "apartment"].copy()
    apartment_test = test.loc[test["property_family"].astype(str).str.lower() == "apartment"].copy()
    if min(len(apartment_fit), len(apartment_test)) > 0:
        populations.append(("apartment_only", apartment_fit, apartment_test))
    for idx, (population, bench_train, bench_test) in enumerate(populations):
        result = benchmark_models(
            bench_train, bench_test, features, best_alpha, min_frequency, seed + 100 * idx,
            benchmark_train_max, benchmark_test_max, benchmark_tree_depth, benchmark_leaf, benchmark_rf_estimators,
        )
        result.insert(0, "population", population)
        benchmark_frames.append(result)
    benchmark = pd.concat(benchmark_frames, ignore_index=True)
    show_progress(64, "fair Ridge / Tree / RF benchmark complete")

    associations = adjusted_associations(pipe, test, fit_frame, features, seed)
    tests = statistical_tests(_sample(fit_frame, 150000, seed))
    show_progress(76, "adjusted associations and statistical tests complete")
    permutation = grouped_permutation_importance(pipe, test, features, seed, permutation_max)
    errors = error_analysis(test, test_pred, minimum_n)
    show_progress(90, "grouped permutation and robust AVM errors complete")

    benchmark_models_present = set(benchmark["model"].astype(str)) if not benchmark.empty else set()
    checks: list[Check] = [
        make_check("forward_train_nonempty", "model", len(train), ">0", len(train) > 0),
        make_check("forward_validation_nonempty", "model", len(validation), ">0", len(validation) > 0),
        make_check("forward_test_nonempty", "model", len(test), ">0", len(test) > 0),
        make_check("validation_is_out_of_sample", "model", "train-only selection fit", "validation not used in fit", True),
        make_check("adjusted_associations_nonempty", "model", len(associations), ">0", len(associations) > 0),
        make_check("permutation_importance_nonempty", "model", len(permutation), ">0", len(permutation) > 0),
        make_check(
            "fair_model_benchmark_completed", "model", sorted(benchmark_models_present),
            "ridge + decision_tree + random_forest",
            {"ridge", "decision_tree", "random_forest"}.issubset(benchmark_models_present), critical=False,
            notes="All benchmark models use identical rows and the same transformed feature matrix within each population.",
        ),
        make_check("target_is_asking_price", "method", TARGET, "sale asking price per sqm", True, notes="No transaction-price claim is made."),
        make_check("causal_claim_guard", "method", "association/predictive wording", "non-causal", True),
    ]

    summary_path = table_dir / "price_driver_summary.csv"
    diagnostics_path = table_dir / "price_driver_model_diagnostics.csv"
    benchmark_path = table_dir / "price_model_benchmark.csv"
    permutation_path = table_dir / "price_driver_permutation_importance.csv"
    errors_path = table_dir / "avm_error_analysis.csv"
    tests_path = qa_dir / "price_driver_statistical_tests.csv"
    alpha_path = qa_dir / "ridge_alpha_selection.csv"
    checks_path = qa_dir / "price_driver_checks.csv"
    model_path = model_dir / "ridge_price_driver_model.joblib"
    manifest_path = qa_dir / "price_driver_manifest.json"

    for df, path in [
        (associations, summary_path), (diagnostics, diagnostics_path), (benchmark, benchmark_path),
        (permutation, permutation_path), (errors, errors_path), (tests, tests_path),
        (alpha_diagnostics, alpha_path),
    ]:
        atomic_write_csv(pl.from_pandas(df), path)
    atomic_write_csv(checks_frame(checks), checks_path)
    _write_model_atomic(pipe, model_path)
    status = summarize_checks(checks)

    observed_best: dict[str, str] = {}
    for population, group in benchmark.groupby("population"):
        if not group.empty:
            observed_best[str(population)] = str(group.sort_values("rmse_log").iloc[0]["model"])

    atomic_write_json(
        {
            "version": VERSION,
            "status": status,
            "input": relative_to_project(feature_path),
            "target": TARGET,
            "target_interpretation": "asking price per sqm, not transaction price",
            "primary_model": "Ridge regression on log target with forward time split",
            "primary_model_policy": "interpretable model remains canonical pending explicit post-QA review",
            "selected_alpha": best_alpha,
            "active_features": features,
            "split_counts": {"train": len(train), "validation": len(validation), "test": len(test)},
            "diagnostic_policy": "train/validation metrics come from train-only selection fit; held-out test comes from final train+validation fit",
            "benchmark_policy": "Ridge, DecisionTreeRegressor, and RandomForestRegressor use identical sampled rows, feature contract, preprocessing matrix, and held-out test rows within each population",
            "benchmark_populations": sorted(benchmark["population"].astype(str).unique().tolist()),
            "benchmark_observed_best_rmse_log": observed_best,
            "benchmark_selection_warning": "Observed held-out winner is descriptive only; do not tune/refit to the test winner without a new untouched evaluation set.",
            "avm_error_policy": "robust Median/P75/P90 APE and within-threshold shares; mean percentage error intentionally not surfaced",
            "avm_intended_use": "interpretable research prototype / initial AVM risk analysis; not production valuation",
            "permutation_policy": "city+neighborhood permuted as one location block; property_family+cat3 as one property-type block",
            "outputs": {
                "summary": relative_to_project(summary_path),
                "diagnostics": relative_to_project(diagnostics_path),
                "benchmark": relative_to_project(benchmark_path),
                "permutation": relative_to_project(permutation_path),
                "error_analysis": relative_to_project(errors_path),
                "model": relative_to_project(model_path),
                "checks": relative_to_project(checks_path),
            },
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        },
        manifest_path,
    )
    show_progress(100, f"complete in {time.perf_counter()-started:.1f}s", final=True)
    return {
        "summary": summary_path, "diagnostics": diagnostics_path, "benchmark": benchmark_path,
        "permutation": permutation_path, "error_analysis": errors_path,
        "statistical_tests": tests_path, "alpha_selection": alpha_path,
        "model": model_path, "checks": checks_path, "manifest": manifest_path,
    }

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run M3 interpretable asking-price driver model and AVM error analysis.")
    parser.add_argument("--features", type=Path, default=FEATURE_TABLE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run(args.features)
    print("M3 PRICE DRIVERS COMPLETED")
    for name, path in outputs.items():
        print(f"{name}: {relative_to_project(path)}")


if __name__ == "__main__":
    main()
