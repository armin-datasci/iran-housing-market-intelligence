from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


CONTRACT_VERSION = "stage2-publication-snapshot-v1"
FOUR_CITIES = {"tehran", "mashhad", "karaj", "isfahan"}


def _first_existing(*paths: Path) -> Path:
    for path in paths:
        if path.is_file():
            return path
    raise FileNotFoundError("No accepted artifact found among: " + ", ".join(str(p) for p in paths))


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported publication-contract table: {path}")


def _num(value: Any) -> float:
    return float(pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0])


def _assert_close(name: str, actual: Any, expected: float, tolerance: float = 1e-6) -> None:
    value = _num(actual)
    if not math.isfinite(value) or not math.isclose(value, expected, rel_tol=0.0, abs_tol=tolerance):
        raise RuntimeError(f"Stage 2 publication snapshot mismatch for {name}: actual={actual!r}, expected={expected!r}")


def _assert_int(name: str, actual: Any, expected: int) -> None:
    value = _num(actual)
    if not math.isfinite(value) or int(round(value)) != int(expected):
        raise RuntimeError(f"Stage 2 publication snapshot mismatch for {name}: actual={actual!r}, expected={expected!r}")


def _require_columns(frame: pd.DataFrame, required: Iterable[str], source: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise RuntimeError(f"Stage 2 publication source {source} is missing columns: {missing}")


def _bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.fillna(False).astype(bool)
    return series.astype("string").str.strip().str.lower().isin({"true", "1", "yes", "pass", "ready"})


def _validate_currency(root: Path) -> dict[str, Any]:
    path = root / "outputs/tables/milestone_2/currency/currency_validation_summary.csv"
    frame = _read_table(path)
    _require_columns(frame, ["stage", "evidence", "population_n", "result", "status"], path.name)
    decision = frame.loc[frame["evidence"].astype(str) == "Currency-scale inference"]
    if len(decision) != 1:
        raise RuntimeError("Stage 2 currency publication requires exactly one Currency-scale inference row.")
    result = str(decision.iloc[0]["result"])
    required_tokens = [
        "scale=1",
        "operational_currency=toman",
        "price_unit=toman_assumed_unconfirmed",
        "factor_of_ten_applied=False",
        "observation_type=asking_price",
    ]
    missing = [token for token in required_tokens if token not in result]
    if missing:
        raise RuntimeError(f"Stage 2 currency decision no longer matches the publication contract: missing {missing}")
    parity = frame.loc[frame["evidence"].astype(str) == "Dataset coverage and Raw/typed parity"]
    if len(parity) == 1:
        parity_result = str(parity.iloc[0]["result"])
        if "raw_typed_mismatches=0" not in parity_result or "monetary_pairs=11" not in parity_result:
            raise RuntimeError("Stage 2 currency parity signature changed; review publication text before release.")
    return {"path": path, "status": str(decision.iloc[0]["status"]).upper()}


def _national_monthly(frame: pd.DataFrame) -> pd.DataFrame:
    _require_columns(frame, ["analysis_month", "entity_level", "deduplicated_listing_count", "median_asking_price_per_sqm_toman"], "monthly market")
    out = frame.copy()
    out["analysis_month"] = pd.to_datetime(out["analysis_month"], errors="coerce")
    out = out.loc[out["entity_level"].astype(str).str.lower() == "national"].copy()
    if "series_kind" in out.columns:
        preferred = out.loc[out["series_kind"].astype(str).str.lower() == "price_and_supply"].copy()
        if not preferred.empty:
            out = preferred
    if "market_scope" in out.columns:
        preferred = out.loc[out["market_scope"].astype(str).isin(["apartment_sale_proxy", "apartment_sale"])].copy()
        if not preferred.empty:
            out = preferred
    out = out.loc[out["analysis_month"].notna()].sort_values("analysis_month")
    if out["analysis_month"].duplicated().any() or len(out) != 8:
        raise RuntimeError("Stage 2 monthly publication signature requires eight unique national apartment-sale months.")
    return out


def _validate_monthly(root: Path) -> dict[str, Any]:
    path = _first_existing(
        root / "data/gold/marts/mart_market_monthly.parquet",
        root / "outputs/tables/milestone_3/monthly_market/monthly_market_summary.csv",
    )
    monthly = _national_monthly(_read_table(path))
    first, last = monthly.iloc[0], monthly.iloc[-1]
    if first["analysis_month"].strftime("%Y-%m") != "2024-05" or last["analysis_month"].strftime("%Y-%m") != "2024-12":
        raise RuntimeError("Stage 2 monthly publication window changed; review the final reports.")
    _assert_int("May supply", first["deduplicated_listing_count"], 34949)
    _assert_int("December supply", last["deduplicated_listing_count"], 41328)
    _assert_close("May median PSM", first["median_asking_price_per_sqm_toman"], 36363636.0, 1.0)
    _assert_close("December median PSM", last["median_asking_price_per_sqm_toman"], 36538462.0, 1.0)
    return {"path": path, "months": len(monthly)}


def _validate_temperature(root: Path) -> dict[str, Any]:
    path = _first_existing(
        root / "data/gold/marts/mart_location_market.parquet",
        root / "outputs/tables/milestone_3/market_temperature/market_temperature_summary.csv",
    )
    frame = _read_table(path)
    _require_columns(frame, ["entity_level", "city_slug", "neighborhood_slug", "market_temperature_label", "market_temperature_score"], path.name)
    work = frame.loc[
        (frame["entity_level"].astype(str).str.lower() == "neighborhood")
        & frame["city_slug"].astype(str).isin(FOUR_CITIES)
        & frame["neighborhood_slug"].notna()
    ].copy()
    reliability = next((c for c in ["temperature_reliability_eligible_flag", "professor_facing_eligible_flag"] if c in work.columns), None)
    if reliability is not None:
        work = work.loc[_bool_series(work[reliability])].copy()
    labels = work["market_temperature_label"].astype(str).str.upper().value_counts().to_dict()
    expected = {"HOT": 17, "COLD": 30, "NEUTRAL": 98}
    for label, count in expected.items():
        if int(labels.get(label, 0)) != count:
            raise RuntimeError(f"Stage 2 Market Temperature publication count changed for {label}: {labels.get(label, 0)} vs {count}")
    return {"path": path, "counts": expected}


def _validate_spatial(root: Path) -> dict[str, Any]:
    path = root / "outputs/tables/milestone_3/spatial_quality/spatial_quality_summary.csv"
    frame = _read_table(path)
    _require_columns(frame, ["metric", "value", "rate", "status"], path.name)
    indexed = frame.set_index(frame["metric"].astype(str))
    expectations = {
        "row_count": (1000000, None),
        "coordinate_pair_rows": (655608, None),
        "iran_window_valid_rows": (655584, None),
        "likely_swapped_rows": (0, None),
        "geo_aggregate_eligible_rows": (655114, None),
        "advanced_boundary_sample_n": (5000, None),
        "advanced_point_in_polygon_inside_rate": (4970, 0.994),
        "reverse_geocode_sample_n": (80, None),
        "reverse_geocode_iran_country_match_rate": (80, 1.0),
        "reverse_geocode_city_or_county_match_rate": (47, 0.5875),
    }
    for metric, (expected_value, expected_rate) in expectations.items():
        if metric not in indexed.index:
            raise RuntimeError(f"Stage 2 spatial publication metric missing: {metric}")
        row = indexed.loc[metric]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        _assert_int(metric, row["value"], expected_value)
        if expected_rate is not None:
            _assert_close(metric + " rate", row["rate"], expected_rate, 1e-6)
    return {"path": path}


def _validate_model(root: Path) -> dict[str, Any]:
    path = _first_existing(
        root / "data/gold/marts/mart_model_quality.parquet",
        root / "outputs/tables/milestone_3/price_drivers/price_driver_model_diagnostics.csv",
    )
    frame = _read_table(path).copy()
    if "evaluation_split" not in frame.columns and "split" in frame.columns:
        frame["evaluation_split"] = frame["split"]
    if "median_ape_pct" not in frame.columns and "median_absolute_percentage_error_pct" in frame.columns:
        frame["median_ape_pct"] = frame["median_absolute_percentage_error_pct"]
    if "record_type" in frame.columns:
        primary = frame.loc[frame["record_type"].astype(str) == "primary_diagnostics"].copy()
        if not primary.empty:
            frame = primary
    _require_columns(frame, ["evaluation_split"], path.name)
    test = frame.loc[frame["evaluation_split"].astype(str).str.lower() == "test"].copy()
    if test.empty:
        raise RuntimeError("Stage 2 model publication requires a held-out test diagnostics row.")
    row = test.iloc[0]
    if "n" in row.index:
        _assert_int("AVM test N", row["n"], 53615)
    if "r2_log" in row.index:
        _assert_close("AVM R2(log)", row["r2_log"], 0.277012, 5e-6)
    if "median_ape_pct" in row.index:
        _assert_close("AVM median APE", row["median_ape_pct"], 33.612925, 5e-4)
    return {"path": path}


def _first_column(frame: pd.DataFrame, candidates: Iterable[str], source: str) -> str:
    for column in candidates:
        if column in frame.columns:
            return column
    raise RuntimeError(
        f"Stage 2 publication source {source} cannot supply any of the accepted columns: {list(candidates)}"
    )


def _validate_seller(root: Path) -> dict[str, Any]:
    # Prefer the accepted Gold mart because it has the stable dashboard/report schema.
    # Fall back to the canonical M3 source so report-only runs remain compatible with
    # older accepted bundles. Both the current compact schema and the earlier verbose
    # analytical schema are intentionally supported.
    summary_path = _first_existing(
        root / "data/gold/marts/mart_seller_type.parquet",
        root / "outputs/tables/milestone_3/seller_type_comparison/seller_type_comparison_summary.csv",
    )
    summary = _read_table(summary_path)
    if summary.empty:
        raise RuntimeError("Stage 2 seller publication source is empty.")

    agency_col = _first_column(summary, ["agency_n", "agency_sample_size"], summary_path.name)
    personal_col = _first_column(summary, ["personal_n", "personal_sample_size"], summary_path.name)
    raw_col = _first_column(summary, ["raw_median_difference_pct", "raw_price_difference_pct"], summary_path.name)
    adjusted_col = _first_column(
        summary, ["adjusted_crossfit_difference_pct", "adjusted_price_difference_pct"], summary_path.name
    )
    ci_low_col = _first_column(summary, ["adjusted_ci_low_pct"], summary_path.name)
    ci_high_col = _first_column(summary, ["adjusted_ci_high_pct"], summary_path.name)

    row = summary.iloc[0]
    _assert_int("seller agency N", row[agency_col], 114843)
    _assert_int("seller personal N", row[personal_col], 11050)
    _assert_close("seller raw gap", row[raw_col], 92.0, 0.15)
    _assert_close("seller adjusted gap", row[adjusted_col], 28.463071, 5e-4)
    _assert_close("seller CI low", row[ci_low_col], 25.973593, 5e-4)
    _assert_close("seller CI high", row[ci_high_col], 31.001747, 5e-4)

    strat_col = next(
        (column for column in ["stratified_difference_pct", "stratified_price_difference_pct"] if column in summary.columns),
        None,
    )
    strat_path: Path | None = None
    if strat_col is not None and pd.notna(row[strat_col]):
        strat_value = row[strat_col]
    else:
        strat_path = root / "outputs/tables/milestone_3/seller_type_comparison/seller_type_stratified_summary.csv"
        strat = _read_table(strat_path)
        if strat.empty:
            raise RuntimeError("Stage 2 seller publication requires the stratified sensitivity summary.")
        strat_col = _first_column(
            strat, ["stratified_difference_pct", "stratified_price_difference_pct"], strat_path.name
        )
        strat_value = strat.iloc[0][strat_col]
    _assert_close("seller stratified gap", strat_value, 17.657784, 5e-4)
    return {"summary": summary_path, "stratified": strat_path or summary_path}


def _validate_text(root: Path) -> dict[str, Any]:
    # Gold exposes a stable compact schema; M3 source artifacts use either the current
    # compact names or the earlier verbose analysis names. Validate the semantic fields
    # rather than pinning publication to one historical column naming convention.
    path = _first_existing(
        root / "data/gold/marts/mart_text_signals.parquet",
        root / "outputs/tables/milestone_3/text_price_signals/text_signal_summary.csv",
        root / "outputs/tables/milestone_3/text_analysis/text_signal_summary.csv",
    )
    frame = _read_table(path)
    if frame.empty:
        raise RuntimeError("Stage 2 text publication source is empty.")

    keyword_col = _first_column(frame, ["keyword_family", "keyword"], path.name)
    precision_col = _first_column(frame, ["manual_precision", "manual_precision_estimate"], path.name)
    effect_col = _first_column(
        frame,
        ["adjusted_effect_pct", "adjusted_residual_difference_pct", "adjusted_price_difference_pct"],
        path.name,
    )
    q_col = _first_column(
        frame, ["q_value", "adjusted_fdr_bh_q_value", "adjusted_p_value_fdr_bh"], path.name
    )

    expected = {
        "new_build": (0.897436, 13.068434, 0.000137),
        "unused": (1.0, 9.393758, 0.028006),
        "urgent": (0.956522, -6.479734, 0.056922),
        "exchange": (0.923077, -11.028695, 0.000234),
        "below_market": (1.0, -4.336094, 0.138758),
        "migration_sale": (1.0, 0.622710, 0.953955),
    }
    by_keyword = frame.set_index(frame[keyword_col].astype(str))
    missing_keywords = set(expected) - set(by_keyword.index)
    if missing_keywords:
        raise RuntimeError(
            f"Stage 2 text publication keyword family changed; missing {sorted(missing_keywords)}. Review the final report."
        )
    for keyword, (precision, effect, q_value) in expected.items():
        row = by_keyword.loc[keyword]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        _assert_close(keyword + " precision", row[precision_col], precision, 5e-6)
        _assert_close(keyword + " adjusted effect", row[effect_col], effect, 5e-4)
        _assert_close(keyword + " q-value", row[q_col], q_value, 5e-6)
    return {"path": path}

def _validate_segments(root: Path) -> dict[str, Any]:
    path = root / "outputs/tables/milestone_3/market_segmentation/segment_profile.csv"
    frame = _read_table(path)
    _require_columns(frame, ["segment_id", "listing_n", "listing_share_pct", "segment_method"], path.name)
    expected_n = {"SEG01": 87065, "SEG02": 145110, "SEG03": 58044, "SEG04": 95119, "SEG05": 20471}
    by_segment = frame.set_index(frame["segment_id"].astype(str))
    if set(by_segment.index) != set(expected_n):
        raise RuntimeError(f"Stage 2 segment IDs changed: {sorted(set(by_segment.index))}")
    for segment_id, expected in expected_n.items():
        row = by_segment.loc[segment_id]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        _assert_int(segment_id + " N", row["listing_n"], expected)
    _assert_int("segment assigned total", pd.to_numeric(frame["listing_n"], errors="coerce").sum(), 405809)
    methods = set(frame["segment_method"].dropna().astype(str))
    allowed = {"rule_based_descriptive_typology", "compatible_domain_segment"}
    if not methods.issubset(allowed):
        raise RuntimeError(f"Stage 2 segment method changed: {sorted(methods)}")
    return {"path": path}


def _validate_governance(root: Path) -> dict[str, Any]:
    method_path = root / "outputs/tables/final_reporting/statistical_method_map.csv"
    design_path = root / "outputs/tables/final_reporting/statistical_design_summary.csv"
    if not method_path.is_file() or not design_path.is_file():
        raise FileNotFoundError("Stage 2 publication requires final statistical-governance tables.")
    return {"method_map": method_path, "design_summary": design_path}


def _validate_gold(root: Path) -> dict[str, Any]:
    path = root / "data/gold/qa/gold_qa_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    status_value = payload.get("status")
    status = status_value if isinstance(status_value, dict) else payload
    overall = str(
        status.get("overall_status", status_value if isinstance(status_value, str) else payload.get("overall_status", ""))
    ).upper()
    critical = int(status.get("critical_failures", payload.get("critical_failures", 0)) or 0)
    if overall != "PASS" or critical != 0:
        raise RuntimeError(f"Stage 2 publication requires Gold PASS with zero critical failures; found {overall}/{critical}")
    return {"path": path, "status": overall}


def validate_stage2_snapshot(root: str | Path) -> dict[str, Any]:
    """Validate the accepted analytical snapshot before publishing the reviewed Stage 2 reports.

    The final Markdown reports are reviewed publication templates tied to the frozen accepted
    analytics release. If an upstream result changes, publication fails instead of silently
    emitting stale narrative numbers; the template must then be reviewed and versioned again.
    """
    project_root = Path(root).resolve()
    checks = {
        "currency": _validate_currency(project_root),
        "monthly": _validate_monthly(project_root),
        "temperature": _validate_temperature(project_root),
        "spatial": _validate_spatial(project_root),
        "model": _validate_model(project_root),
        "seller": _validate_seller(project_root),
        "text": _validate_text(project_root),
        "segments": _validate_segments(project_root),
        "governance": _validate_governance(project_root),
        "gold": _validate_gold(project_root),
    }
    return {"contract_version": CONTRACT_VERSION, "checks": checks}
