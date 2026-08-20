from __future__ import annotations

import argparse
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

from src.final_reporting.publication_contract import validate_stage2_snapshot


REPORT_VERSION = "final-executive-summary-v2.4-stage2-complete"
DEFAULT_OUTPUT = Path("reports/final/Executive_Summary.md")
TEMPERATURE_LEVEL = "neighborhood"


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
    text = series.astype("string").str.strip().str.lower()
    return text.isin({"true", "1", "yes", "pass", "passed", "ready"})


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([float("inf"), float("-inf")], pd.NA)


def fmt_int(value: Any) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{int(round(float(value))):,}"


def fmt_num(value: Any, digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):,.{digits}f}"


def fmt_pct(value: Any, digits: int = 1, signed: bool = False) -> str:
    if value is None or pd.isna(value):
        return "-"
    number = float(value)
    spec = f"+,.{digits}f" if signed else f",.{digits}f"
    return f"{format(number, spec)}%"


def fmt_toman(value: Any) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):,.0f} Toman"


def slug_label(value: Any) -> str:
    if value is None or pd.isna(value):
        return "-"
    text = str(value).strip()
    if not text:
        return "-"
    return " ".join(part.capitalize() for part in re.split(r"[-_]+", text) if part)


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
        raise RuntimeError(f"Executive Summary contains non-English script near: {excerpt!r}")


def _assert_no_personal_paths(text: str) -> None:
    patterns = [r"[A-Za-z]:\\Users\\", r"/Users/[^/]+/", r"/home/[^/]+/"]
    for pattern in patterns:
        if re.search(pattern, text):
            raise RuntimeError("Executive Summary contains a personal filesystem path.")


def _gold_ready(root: Path) -> tuple[bool | None, str, Path | None]:
    path = root / "data" / "gold" / "qa" / "gold_qa_manifest.json"
    if not path.exists():
        return None, "Gold QA manifest not found.", None
    payload = read_json(path)
    ready_flag = bool(payload.get("gold_data_contract_ready"))
    status = payload.get("status") if isinstance(payload.get("status"), dict) else {}
    architecture = payload.get("architecture") if isinstance(payload.get("architecture"), dict) else {}
    overall = str(status.get("overall_status", "UNKNOWN")).upper()
    critical = int(status.get("critical_failures") or 0)
    architecture_ok = (
        architecture.get("gold_marts") == 10
        and architecture.get("conformed_dimensions") == 5
        and architecture.get("physical_relationships") == 13
        and architecture.get("semantic_only_dimensions") in (["dim_user_type"], None)
    )
    effective_ready = bool(ready_flag and overall == "PASS" and critical == 0 and architecture_ok)
    note = (
        f"Gold QA={overall}; critical failures={critical}; data contract ready={ready_flag}; "
        f"architecture=10 marts/5 dimensions/13 physical relationships."
    )
    return effective_ready, note, path


def _candidate_paths(root: Path) -> dict[str, tuple[Path, ...]]:
    gold = root / "data" / "gold"
    marts = gold / "marts"
    dims = gold / "dimensions"
    m3 = root / "outputs" / "tables" / "milestone_3"
    m3_qa = root / "outputs" / "qa" / "milestone_3"
    return {
        "temperature_summary": (
            m3 / "market_temperature" / "market_temperature_summary.csv",
        ),
        "temperature_manifest": (
            m3_qa / "market_temperature" / "market_temperature_manifest.json",
            m3 / "market_temperature" / "market_temperature_manifest.json",
        ),
        "dim_location": (
            dims / "dim_location.parquet",
            dims / "dim_location.csv",
        ),
        "gold_location": (
            marts / "mart_location_market.parquet",
            marts / "mart_location_market.csv",
        ),
        "monthly": (
            marts / "mart_market_monthly.parquet",
            marts / "mart_market_monthly.csv",
            m3 / "monthly_market" / "monthly_market_summary.csv",
        ),
        "effects": (
            marts / "mart_price_driver_effects.parquet",
            marts / "mart_price_driver_effects.csv",
            m3 / "price_drivers" / "price_driver_summary.csv",
        ),
        "importance": (
            marts / "mart_price_driver_importance.parquet",
            marts / "mart_price_driver_importance.csv",
            m3 / "price_drivers" / "price_driver_permutation_importance.csv",
        ),
        "model_quality": (
            marts / "mart_model_quality.parquet",
            marts / "mart_model_quality.csv",
            m3 / "price_drivers" / "price_driver_model_diagnostics.csv",
        ),
    }


def _load_required_path(root: Path, key: str, options: Sequence[Path]) -> Path:
    path = first_existing(*options)
    if path is None:
        rendered = " | ".join(relative_path(item, root) for item in options)
        raise FileNotFoundError(f"Required Executive Summary source is missing ({key}): {rendered}")
    return path


def _load_first_compatible(
    root: Path,
    key: str,
    options: Sequence[Path],
    required_columns: set[str],
) -> tuple[pd.DataFrame, Path]:
    errors: list[str] = []
    for path in options:
        if not path.exists():
            continue
        try:
            frame = read_table(path)
        except Exception as exc:
            errors.append(f"{relative_path(path, root)} read error: {exc}")
            continue
        missing = sorted(required_columns - set(frame.columns))
        if not missing:
            return frame, path
        errors.append(f"{relative_path(path, root)} missing columns: {missing}")
    detail = "\n- ".join(errors) if errors else "No candidate file exists."
    raise FileNotFoundError(f"No compatible source found for {key}.\n- {detail}")


def _four_city_slugs(dim_location: pd.DataFrame) -> set[str]:
    require_columns(dim_location, ["city_slug", "is_four_city"], "dim_location")
    work = dim_location.copy()
    if "location_level" in work.columns:
        city_rows = work.loc[work["location_level"].astype(str).str.lower() == "city"].copy()
        if not city_rows.empty:
            work = city_rows
    selected = work.loc[bool_series(work["is_four_city"]), "city_slug"].dropna().astype(str).str.strip()
    cities = {value for value in selected if value}
    if len(cities) != 4:
        raise ValueError(f"dim_location.is_four_city must identify exactly four cities; found {sorted(cities)}")
    return cities


def _unique_int_from_column(frame: pd.DataFrame, names: Sequence[str]) -> int | None:
    for name in names:
        if name not in frame.columns:
            continue
        values = numeric(frame[name]).dropna().astype(int).unique().tolist()
        if len(values) == 1:
            return int(values[0])
        if len(values) > 1:
            raise ValueError(f"Expected one reliability threshold in {name}; found {values}")
    return None


def _temperature_gate(
    manifest_path: Path | None,
    gold_location: pd.DataFrame,
) -> tuple[int, int, int, str]:
    gold_price = _unique_int_from_column(
        gold_location,
        ["temperature_minimum_price_months", "professor_facing_minimum_price_months"],
    )
    gold_activity = _unique_int_from_column(
        gold_location,
        ["temperature_minimum_activity_months", "professor_facing_minimum_activity_months"],
    )
    gold_n = _unique_int_from_column(
        gold_location,
        ["temperature_minimum_n", "professor_facing_minimum_n"],
    )
    gold_values = (gold_price, gold_activity, gold_n)

    manifest_values: tuple[int, int, int] | None = None
    if manifest_path is not None:
        payload = read_json(manifest_path)
        definition = payload.get("definition") if isinstance(payload.get("definition"), dict) else {}
        gate = (
            definition.get("professor_facing_reliability_gate")
            if isinstance(definition.get("professor_facing_reliability_gate"), dict)
            else {}
        )
        raw_values = (
            gate.get("minimum_reliable_price_months"),
            gate.get("minimum_listing_activity_months"),
            gate.get("minimum_sample_size"),
        )
        if all(value is not None for value in raw_values):
            manifest_values = tuple(int(value) for value in raw_values)  # type: ignore[arg-type]

    if all(value is not None for value in gold_values):
        resolved = tuple(int(value) for value in gold_values)  # type: ignore[arg-type]
        if manifest_values is not None and resolved != manifest_values:
            raise RuntimeError(
                f"Market-temperature reliability gate mismatch between Gold {resolved} and M3 manifest {manifest_values}."
            )
        return resolved[0], resolved[1], resolved[2], "Gold temperature threshold columns"

    if manifest_values is not None:
        return manifest_values[0], manifest_values[1], manifest_values[2], manifest_path.as_posix()

    raise RuntimeError(
        "Professor-facing market-temperature reliability thresholds could not be resolved from Gold or the canonical M3 manifest."
    )


def _prepare_presentation_temperature(
    gold_location: pd.DataFrame,
    four_city_slugs: set[str],
    minimum_price_months: int,
    minimum_activity_months: int,
    minimum_n: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    # Final M4 location mart intentionally renames the upstream temperature sample
    # from sample_size -> temperature_sample_n before merging it with Price Map.
    # The generic Gold sample_n is NOT safe here because, on mapped neighborhood
    # rows, it can represent the Price Map valid-price N instead of temperature N.
    require_columns(
        gold_location,
        [
            "entity_level",
            "city_slug",
            "neighborhood_slug",
            "price_trend_pct_per_month",
            "listing_activity_trend_pct_per_month",
            "price_month_count",
            "listing_activity_month_count",
            "market_temperature_score",
            "market_temperature_label",
        ],
        "mart_location_market",
    )

    work = gold_location.copy()
    if "temperature_sample_n" in work.columns:
        work["temperature_sample_n"] = numeric(work["temperature_sample_n"])
        if "sample_size" in work.columns:
            legacy_sample = numeric(work["sample_size"])
            overlap = work["temperature_sample_n"].notna() & legacy_sample.notna()
            if overlap.any() and not (work.loc[overlap, "temperature_sample_n"].reset_index(drop=True) == legacy_sample.loc[overlap].reset_index(drop=True)).all():
                raise RuntimeError(
                    "mart_location_market contains conflicting temperature_sample_n and legacy sample_size values."
                )
    elif "sample_size" in work.columns:
        # Compatibility only for the pre-merge M4 temperature mart.
        work["temperature_sample_n"] = numeric(work["sample_size"])
    else:
        raise ValueError(
            "mart_location_market is missing temperature_sample_n (the final M4 temperature sample field)."
        )

    reliability_column = (
        "temperature_reliability_eligible_flag"
        if "temperature_reliability_eligible_flag" in work.columns
        else "professor_facing_eligible_flag"
        if "professor_facing_eligible_flag" in work.columns
        else None
    )
    if reliability_column is None:
        raise ValueError(
            "mart_location_market is missing temperature_reliability_eligible_flag."
        )

    for column in [
        "market_temperature_score",
        "price_trend_pct_per_month",
        "listing_activity_trend_pct_per_month",
        "price_month_count",
        "listing_activity_month_count",
        "temperature_sample_n",
    ]:
        work[column] = numeric(work[column])
    work["market_temperature_label"] = work["market_temperature_label"].astype(str).str.upper()

    presentation = work.loc[
        (work["entity_level"].astype(str).str.lower() == TEMPERATURE_LEVEL)
        & work["city_slug"].astype(str).isin(four_city_slugs)
        & work["neighborhood_slug"].notna()
        & (work["neighborhood_slug"].astype(str).str.strip() != "")
        & bool_series(work[reliability_column])
    ].copy()

    if presentation.empty:
        raise ValueError("The four-city reliable neighborhood market-temperature presentation pool is empty in Gold.")

    reliability_violations = presentation.loc[
        (presentation["price_month_count"] < minimum_price_months)
        | (presentation["listing_activity_month_count"] < minimum_activity_months)
        | (presentation["temperature_sample_n"] < minimum_n)
    ]
    if not reliability_violations.empty:
        raise RuntimeError(
            f"Gold temperature reliability eligibility contains {len(reliability_violations)} row(s) that violate the published reliability gate."
        )

    hot_pool = presentation.loc[presentation["market_temperature_label"] == "HOT"].copy()
    cold_pool = presentation.loc[presentation["market_temperature_label"] == "COLD"].copy()
    neutral_pool = presentation.loc[presentation["market_temperature_label"] == "NEUTRAL"].copy()

    hot_direction_violations = hot_pool.loc[~(hot_pool["price_trend_pct_per_month"] > 0)].shape[0]
    cold_direction_violations = cold_pool.loc[~(cold_pool["price_trend_pct_per_month"] < 0)].shape[0]
    if hot_direction_violations or cold_direction_violations:
        raise RuntimeError(
            "Market-temperature direction gate mismatch: HOT must have positive asking-price trend and COLD negative asking-price trend."
        )

    hot = hot_pool.nlargest(5, "market_temperature_score").copy()
    cold = cold_pool.nsmallest(5, "market_temperature_score").copy()
    if len(hot) != 5 or len(cold) != 5:
        raise RuntimeError(
            f"Expected five reliable HOT and five reliable COLD neighborhoods; got HOT={len(hot)}, COLD={len(cold)}."
        )

    counts = {"HOT": len(hot_pool), "COLD": len(cold_pool), "NEUTRAL": len(neutral_pool)}
    return hot, cold, counts


def _reconcile_temperature_with_m3(
    hot: pd.DataFrame,
    cold: pd.DataFrame,
    source: pd.DataFrame,
) -> None:
    require_columns(
        source,
        [
            "entity_level", "city_slug", "neighborhood_slug", "market_temperature_score",
            "market_temperature_label", "price_trend_pct_per_month", "supply_trend_pct_per_month",
            "price_month_count", "supply_month_count", "sample_size",
        ],
        "market_temperature_summary.csv",
    )
    m3 = source.copy()
    m3 = m3.loc[m3["entity_level"].astype(str).str.lower() == "neighborhood"].copy()
    m3["market_temperature_score"] = numeric(m3["market_temperature_score"])
    m3["market_temperature_label"] = m3["market_temperature_label"].astype(str).str.upper()
    selected = pd.concat([hot, cold], ignore_index=True)

    for _, row in selected.iterrows():
        match = m3.loc[
            (m3["city_slug"].astype(str) == str(row["city_slug"]))
            & (m3["neighborhood_slug"].astype(str) == str(row["neighborhood_slug"]))
        ]
        if len(match) != 1:
            raise RuntimeError(
                f"M3 reconciliation expected one row for {row['city_slug']}/{row['neighborhood_slug']}; found {len(match)}."
            )
        m3_row = match.iloc[0]
        if str(m3_row["market_temperature_label"]).upper() != str(row["market_temperature_label"]).upper():
            raise RuntimeError(f"M3/Gold temperature label mismatch for {row['city_slug']}/{row['neighborhood_slug']}.")
        numeric_pairs = [
            ("market_temperature_score", "market_temperature_score"),
            ("price_trend_pct_per_month", "price_trend_pct_per_month"),
            ("listing_activity_trend_pct_per_month", "supply_trend_pct_per_month"),
            ("price_month_count", "price_month_count"),
            ("listing_activity_month_count", "supply_month_count"),
            ("temperature_sample_n", "sample_size"),
        ]
        for gold_column, m3_column in numeric_pairs:
            gold_value = float(row[gold_column])
            m3_value = float(m3_row[m3_column])
            if not math.isclose(gold_value, m3_value, rel_tol=1e-9, abs_tol=1e-6):
                raise RuntimeError(
                    f"M3/Gold {gold_column} mismatch for {row['city_slug']}/{row['neighborhood_slug']}."
                )


def _national_monthly(frame: pd.DataFrame) -> pd.DataFrame:
    require_columns(frame, ["analysis_month", "entity_level", "deduplicated_listing_count"], "monthly market")
    out = frame.copy()
    if "median_asking_price_per_sqm_toman" not in out.columns:
        raise ValueError("Monthly market source lacks median_asking_price_per_sqm_toman.")
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
    if out.empty:
        raise ValueError("No national apartment-sale monthly series is available.")
    if out["analysis_month"].duplicated().any():
        duplicate_months = out.loc[out["analysis_month"].duplicated(keep=False), "analysis_month"].dt.strftime("%Y-%m").tolist()
        raise ValueError(f"National monthly series is not unique by month: {duplicate_months}")
    return out


def _monthly_summary(frame: pd.DataFrame) -> dict[str, Any]:
    monthly = _national_monthly(frame)
    first = monthly.iloc[0]
    last = monthly.iloc[-1]
    first_supply = float(first["deduplicated_listing_count"])
    last_supply = float(last["deduplicated_listing_count"])
    first_price = float(first["median_asking_price_per_sqm_toman"])
    last_price = float(last["median_asking_price_per_sqm_toman"])
    supply_change = ((last_supply / first_supply) - 1.0) * 100.0 if first_supply else None
    price_change = ((last_price / first_price) - 1.0) * 100.0 if first_price else None

    supply_mom_col = "deduplicated_supply_mom_pct"
    price_mom_col = "median_price_mom_pct"
    median_supply_mom = numeric(monthly[supply_mom_col]).dropna().median() if supply_mom_col in monthly.columns else None
    median_price_mom = numeric(monthly[price_mom_col]).dropna().median() if price_mom_col in monthly.columns else None

    return {
        "start_month": first["analysis_month"],
        "end_month": last["analysis_month"],
        "start_supply": first_supply,
        "end_supply": last_supply,
        "supply_change_pct": supply_change,
        "median_supply_mom_pct": median_supply_mom,
        "start_price": first_price,
        "end_price": last_price,
        "price_change_pct": price_change,
        "median_price_mom_pct": median_price_mom,
        "month_count": int(monthly["analysis_month"].nunique()),
    }


def _normalize_importance(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "feature_or_block_id" not in out.columns and "feature_block" in out.columns:
        out["feature_or_block_id"] = out["feature_block"]
    if "feature_or_block_name" not in out.columns:
        out["feature_or_block_name"] = out["feature_or_block_id"].map(slug_label)
    if "permutation_importance" not in out.columns and "heldout_rmse_log_increase_mean" in out.columns:
        out["permutation_importance"] = out["heldout_rmse_log_increase_mean"]
    require_columns(out, ["feature_or_block_id", "feature_or_block_name", "permutation_importance"], "price-driver importance")
    out["permutation_importance"] = numeric(out["permutation_importance"])
    return out.sort_values("permutation_importance", ascending=False)


def _normalize_effects(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "feature_id" not in out.columns and "feature" in out.columns:
        out["feature_id"] = out["feature"]
    if "feature_display_name" not in out.columns:
        out["feature_display_name"] = out["feature_id"].map(slug_label)
    if "contrast_definition" not in out.columns:
        out["contrast_definition"] = "-"
    require_columns(out, ["feature_id", "feature_display_name", "adjusted_effect_pct"], "price-driver effects")
    out["adjusted_effect_pct"] = numeric(out["adjusted_effect_pct"])
    out = out.loc[~out["feature_id"].astype(str).str.contains("unselect", case=False, na=False)].copy()
    out["abs_effect"] = out["adjusted_effect_pct"].abs()
    return out.sort_values("abs_effect", ascending=False)


def _model_test_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    out = frame.copy()
    if "evaluation_split" not in out.columns and "split" in out.columns:
        out["evaluation_split"] = out["split"]
    if "median_ape_pct" not in out.columns and "median_absolute_percentage_error_pct" in out.columns:
        out["median_ape_pct"] = out["median_absolute_percentage_error_pct"]
    if "within_30_pct" not in out.columns and "within_30pct_share_pct" in out.columns:
        out["within_30_pct"] = out["within_30pct_share_pct"]
    if "median_absolute_error_psm_toman" not in out.columns and "median_abs_error_price_per_sqm_toman" in out.columns:
        out["median_absolute_error_psm_toman"] = out["median_abs_error_price_per_sqm_toman"]
    if "record_type" in out.columns:
        primary = out.loc[out["record_type"].astype(str) == "primary_diagnostics"].copy()
        if not primary.empty:
            out = primary
    if "evaluation_split" not in out.columns:
        return {}
    test = out.loc[out["evaluation_split"].astype(str).str.lower() == "test"].copy()
    if test.empty:
        return {}
    row = test.iloc[0]
    return {
        "n": row.get("n"),
        "r2_log": row.get("r2_log"),
        "median_ape_pct": row.get("median_ape_pct"),
        "within_30_pct": row.get("within_30_pct"),
        "median_absolute_error_psm_toman": row.get("median_absolute_error_psm_toman"),
    }


def _market_name(row: pd.Series) -> str:
    return f"{slug_label(row.get('city_slug'))} / {slug_label(row.get('neighborhood_slug'))}"


def build_executive_summary(
    project_root: str | Path | None = None,
    output_path: str | Path | None = None,
    *,
    require_gold_ready: bool = True,
) -> Path:
    root = resolve_project_root(project_root)
    output = Path(output_path) if output_path is not None else root / DEFAULT_OUTPUT
    if not output.is_absolute():
        output = root / output

    gold_ready, gold_note, gold_manifest_path = _gold_ready(root)
    if require_gold_ready and gold_ready is not True:
        raise RuntimeError(f"Executive Summary generation requires ready Gold. {gold_note}")

    candidates = _candidate_paths(root)
    temperature_path = _load_required_path(root, "temperature_summary", candidates["temperature_summary"])
    dim_location_path = _load_required_path(root, "dim_location", candidates["dim_location"])
    gold_location_path = _load_required_path(root, "gold_location", candidates["gold_location"])
    temperature = read_table(temperature_path)
    dim_location = read_table(dim_location_path)
    gold_location = read_table(gold_location_path)
    four_cities = _four_city_slugs(dim_location)

    temperature_manifest_path = first_existing(*candidates["temperature_manifest"])
    min_price_months, min_activity_months, min_n, gate_source = _temperature_gate(
        temperature_manifest_path,
        gold_location,
    )
    hot, cold, temp_counts = _prepare_presentation_temperature(
        gold_location,
        four_cities,
        min_price_months,
        min_activity_months,
        min_n,
    )
    _reconcile_temperature_with_m3(hot, cold, temperature)

    monthly_frame, monthly_path = _load_first_compatible(
        root,
        "monthly",
        candidates["monthly"],
        {"analysis_month", "entity_level", "deduplicated_listing_count", "median_asking_price_per_sqm_toman"},
    )
    effects_frame, effects_path = _load_first_compatible(root, "effects", candidates["effects"], {"adjusted_effect_pct"})
    importance_frame, importance_path = _load_first_compatible(root, "importance", candidates["importance"], set())
    model_frame, model_path = _load_first_compatible(root, "model_quality", candidates["model_quality"], set())

    monthly = _monthly_summary(monthly_frame)
    importance = _normalize_importance(importance_frame)
    effects = _normalize_effects(effects_frame)
    model_metrics = _model_test_metrics(model_frame)

    hot_rows = [
        [
            _market_name(row),
            fmt_num(row.get("market_temperature_score"), 1),
            fmt_pct(row.get("price_trend_pct_per_month"), 2, signed=True),
            fmt_pct(row.get("listing_activity_trend_pct_per_month"), 2, signed=True),
            fmt_int(row.get("temperature_sample_n")),
        ]
        for _, row in hot.iterrows()
    ]
    cold_rows = [
        [
            _market_name(row),
            fmt_num(row.get("market_temperature_score"), 1),
            fmt_pct(row.get("price_trend_pct_per_month"), 2, signed=True),
            fmt_pct(row.get("listing_activity_trend_pct_per_month"), 2, signed=True),
            fmt_int(row.get("temperature_sample_n")),
        ]
        for _, row in cold.iterrows()
    ]

    importance_rows: list[list[str]] = []
    for _, row in importance.head(6).iterrows():
        feature_id = str(row.get("feature_or_block_id"))
        role = (
            "location control" if feature_id == "location"
            else "property-type control" if feature_id == "property_type"
            else "time control" if feature_id == "time"
            else "property characteristic"
        )
        importance_rows.append([str(row.get("feature_or_block_name")), role, fmt_num(row.get("permutation_importance"), 4)])

    effects_nonzero = effects.loc[effects["abs_effect"].fillna(0) > 1e-12].head(5)
    effect_rows = [
        [
            str(row.get("feature_display_name")),
            str(row.get("contrast_definition", "-")),
            fmt_pct(row.get("adjusted_effect_pct"), 1, signed=True),
        ]
        for _, row in effects_nonzero.iterrows()
    ]

    start_month = pd.Timestamp(monthly["start_month"]).strftime("%Y-%m")
    end_month = pd.Timestamp(monthly["end_month"]).strftime("%Y-%m")
    model_limitation = ""
    if model_metrics:
        model_limitation = (
            f" On the held-out test set, the Ridge model had R2(log)={fmt_num(model_metrics.get('r2_log'), 3)} "
            f"and median absolute percentage error={fmt_pct(model_metrics.get('median_ape_pct'), 1)}; "
            "it is therefore more appropriate for approximate prediction and association analysis than point-precise valuation."
        )

    generated_at = datetime.now(timezone.utc).isoformat()
    city_list = ", ".join(sorted(slug_label(city) for city in four_cities))
    text = f"""# Executive Summary - Iran Housing Market Intelligence

- **Report version:** `{REPORT_VERSION}`
- **Generated at (UTC):** `{generated_at}`
- **Core analytical period:** `{start_month}` to `{end_month}` ({monthly['month_count']} months)
- **Price observation:** listing asking price, not transaction price
- **Operational currency:** Toman (`toman_assumed_unconfirmed`)
- **Gold status:** {gold_note}

## 1. Overall market status

The final analysis uses the canonical Silver Master, accepted analytical outputs, and the validated Gold data layer. In the national apartment-sale series, deduplicated listing activity increased from **{fmt_int(monthly['start_supply'])}** listings in `{start_month}` to **{fmt_int(monthly['end_supply'])}** in `{end_month}`, a cumulative change of **{fmt_pct(monthly['supply_change_pct'], 1, signed=True)}**. The median valid month-over-month change in deduplicated listing activity was **{fmt_pct(monthly['median_supply_mom_pct'], 2, signed=True)}**.

Over the same period, the national median apartment asking price per square meter moved from **{fmt_toman(monthly['start_price'])}** to **{fmt_toman(monthly['end_price'])}**, a cumulative change of **{fmt_pct(monthly['price_change_pct'], 1, signed=True)}**. The median valid month-over-month asking-price change was **{fmt_pct(monthly['median_price_mom_pct'], 2, signed=True)}**. These values describe listing behavior, not completed transactions.

Market Temperature is a **Listing-Market Temperature Proxy** based on asking-price trend and listing-activity trend. The canonical ranking remains all-city within each entity level, while the executive top/bottom tables reproduce the project's professor-facing reliable neighborhood view for **{city_list}**. The reliability gate is at least **{min_price_months} price months**, **{min_activity_months} listing-activity months**, and **N >= {min_n}**. In this presentation pool there are **{temp_counts['HOT']} HOT**, **{temp_counts['COLD']} COLD**, and **{temp_counts['NEUTRAL']} NEUTRAL** neighborhoods.

## 2. Five hottest neighborhood markets

{_md_table(['Market', 'Temperature score', 'Asking-price trend / month', 'Listing-activity trend / month', 'N'], hot_rows)}

HOT indicates a positive relative proxy signal after the reliability gate. It does not mean highest liquidity, best investment return, or strongest physical demand.

## 3. Five coldest neighborhood markets

{_md_table(['Market', 'Temperature score', 'Asking-price trend / month', 'Listing-activity trend / month', 'N'], cold_rows)}

COLD indicates a negative relative proxy signal in the listing data. It should not be interpreted by itself as structural market weakness.

## 4. Supply and asking-price trends

- National deduplicated apartment-sale listing activity over the core period: **{fmt_pct(monthly['supply_change_pct'], 1, signed=True)}**.
- National median apartment asking price per square meter over the core period: **{fmt_pct(monthly['price_change_pct'], 1, signed=True)}**.
- Median valid month-over-month listing-activity change: **{fmt_pct(monthly['median_supply_mom_pct'], 2, signed=True)}**.
- Median valid month-over-month asking-price change: **{fmt_pct(monthly['median_price_mom_pct'], 2, signed=True)}**.
- Listing counts are a platform-activity measure; they are not physical housing inventory or transaction volume.

## 5. Most important factors associated with asking price

### 5.1 Held-out predictive contribution

{_md_table(['Feature / block', 'Role', 'Permutation increase in RMSE(log)'], importance_rows)}

Permutation importance measures **held-out predictive contribution**, not causal importance. Location and property-type blocks are structural controls and should be separated from potentially actionable property characteristics.

### 5.2 Model-implied adjusted associations

{_md_table(['Property characteristic', 'Contrast', 'Adjusted asking-price association'], effect_rows) if effect_rows else 'No non-zero reportable contrasts were available in the canonical output.'}

Adjusted effects are **model-implied associations/contrasts**. They are not shares of price and they are not causal effects.

## 6. Main limitations

1. **Asking-price data:** observed prices are listing asks, not realized transaction prices.
2. **Currency status:** Toman is the project's operational unit, but the source currency is not independently confirmed; no undocumented factor-of-ten conversion is applied.
3. **Platform selection bias:** the dataset is not the complete Iranian housing stock or transaction universe.
4. **Market Temperature semantics:** the index combines asking-price and listing-activity trends; it is not liquidity, absorption, physical inventory, or supply tightness.
5. **Model interpretation:** price-driver results are observational and non-causal.{model_limitation}
6. **Spatial privacy:** exact coordinates are not exposed in Gold or the dashboard; spatial outputs are aggregated.
7. **Presentation scope:** the five HOT/COLD tables are the reliable four-city professor-facing view, while the underlying temperature model and canonical labels are computed on the broader eligible ranking universe.

## 7. Management takeaway

The core-period listing market is heterogeneous rather than uniformly hot or cold. National apartment-sale listing activity increased over the period, while the national median asking price per square meter changed much less. Local differences are substantial, so HOT/COLD rankings should be read together with sample size, asking-price trend, listing-activity trend, and reliability conditions. Predictive modeling confirms that location and structural property characteristics matter for prediction, but the evidence does not by itself establish causal price effects or investment recommendations.

---

### Canonical sources used

- `temperature_selection_gold`: `{relative_path(gold_location_path, root)}`
- `temperature_source_reconciliation`: `{relative_path(temperature_path, root)}`
- `dim_location`: `{relative_path(dim_location_path, root)}`
- `temperature_gate`: `{relative_path(temperature_manifest_path, root) if temperature_manifest_path is not None else gate_source}`
- `monthly_market`: `{relative_path(monthly_path, root)}`
- `price_driver_importance`: `{relative_path(importance_path, root)}`
- `price_driver_effects`: `{relative_path(effects_path, root)}`
- `model_quality`: `{relative_path(model_path, root)}`
"""
    if gold_manifest_path is not None:
        text += f"- `gold_qa`: `{relative_path(gold_manifest_path, root)}`\n"
    text += "\nThis report builder does not refit models or recompute upstream analytical estimates; it summarizes accepted canonical outputs.\n"

    # The reviewed report in reports/final is the single publication source.
    # Upstream artifacts are validated first; the report is never regenerated from
    # a duplicate template under src/. If analytics drift, the publication contract
    # fails and the report must be reviewed/versioned explicitly.
    validate_stage2_snapshot(root)
    if not output.is_file():
        raise FileNotFoundError(f"Reviewed final Executive Summary not found: {output}")
    publication_text = output.read_text(encoding="utf-8-sig")
    if REPORT_VERSION not in publication_text:
        raise RuntimeError(
            f"Executive Summary version mismatch: expected {REPORT_VERSION!r} in {output}"
        )
    _assert_english_only(publication_text)
    _assert_no_personal_paths(publication_text)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the reviewed final English Executive Summary against canonical IHMI artifacts.")
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--allow-unready-gold", action="store_true", help="Allow report generation when Gold QA is not ready (not recommended for final delivery).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    path = build_executive_summary(
        project_root=args.project_root,
        output_path=args.output,
        require_gold_ready=not args.allow_unready_gold,
    )
    print(f"EXECUTIVE SUMMARY VALIDATED: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
