from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Iterable

import polars as pl

from src.common.config import price_observation_type, price_unit, setting
from src.common.domain import load_domain_mappings
from src.common.io_utils import atomic_sink_parquet, atomic_write_csv, atomic_write_json
from src.common.paths import DATA_DIR, OUTPUTS_DIR, relative_to_project
from src.common.validation import Check, checks_frame, make_check, summarize_checks
from src.milestone_4.gold.contracts import (
    COMMON_MART_METADATA_COLUMNS,
    DASHBOARD_METADATA_FILENAMES,
    DISCONNECTED_DIMENSIONS,
    SEMANTIC_ONLY_DIMENSIONS,
    EXPECTED_DIMENSIONS,
    EXPECTED_MARTS,
    EXPECTED_TEXT_SIGNALS,
    FORBIDDEN_COLUMN_TOKENS,
    GOLD_BUILD_VERSION,
    LEGACY_ARTIFACT_NAMES,
    MART_GRAIN_KEYS,
    RELATIONSHIP_SPECS,
    REQUIRED_ARTIFACT_COLUMNS,
    artifact_contract_rows,
    dashboard_page_rows,
    relationship_contract_rows,
)

VERSION = GOLD_BUILD_VERSION
PROGRESS_WIDTH = 30
FOUR_CITY_SLUGS = ("tehran", "mashhad", "karaj", "isfahan")

GOLD_ROOT = DATA_DIR / "gold"
MARTS_DIR = GOLD_ROOT / "marts"
DIMENSIONS_DIR = GOLD_ROOT / "dimensions"
METADATA_DIR = GOLD_ROOT / "metadata"
QA_DIR = GOLD_ROOT / "qa"

M3_TABLES = OUTPUTS_DIR / "tables" / "milestone_3"
M3_QA = OUTPUTS_DIR / "qa" / "milestone_3"
M3_MODELS = OUTPUTS_DIR / "model_artifacts" / "milestone_3"

def first_existing_path(*paths: Path) -> Path:
    """Return the first existing candidate, or the canonical first candidate.

    The first path is always the post-refactor canonical location. Additional
    candidates only preserve compatibility with already-produced frozen M3
    manifests; they do not reintroduce legacy analytical artifacts.
    """
    if not paths:
        raise ValueError("At least one path candidate is required.")
    for path in paths:
        if path.exists():
            return path
    return paths[0]


SOURCES = {
    "monthly_market": M3_TABLES / "monthly_market" / "monthly_market_summary.csv",
    "monthly_supply": M3_TABLES / "monthly_market" / "monthly_supply_summary.csv",
    "monthly_manifest": first_existing_path(
        M3_QA / "monthly_market" / "monthly_market_manifest.json",
        M3_QA / "monthly_market" / "monthly_market_summary_manifest.json",
        M3_TABLES / "monthly_market" / "monthly_market_summary_manifest.json",
    ),
    "neighborhood_market": M3_TABLES / "market_map" / "neighborhood_market_summary.csv",
    "market_map_manifest": first_existing_path(
        M3_QA / "market_map" / "market_map_manifest.json",
        M3_TABLES / "market_map" / "market_map_manifest.json",
    ),
    "market_temperature": M3_TABLES / "market_temperature" / "market_temperature_summary.csv",
    "market_temperature_manifest": first_existing_path(
        M3_QA / "market_temperature" / "market_temperature_manifest.json",
        M3_TABLES / "market_temperature" / "market_temperature_manifest.json",
    ),
    "price_driver_effects": M3_TABLES / "price_drivers" / "price_driver_summary.csv",
    "price_driver_diagnostics": M3_TABLES / "price_drivers" / "price_driver_model_diagnostics.csv",
    "price_driver_benchmark": M3_TABLES / "price_drivers" / "price_model_benchmark.csv",
    "price_driver_importance": M3_TABLES / "price_drivers" / "price_driver_permutation_importance.csv",
    "avm_error_analysis": M3_TABLES / "price_drivers" / "avm_error_analysis.csv",
    "price_driver_manifest": first_existing_path(
        M3_QA / "price_drivers" / "price_driver_manifest.json",
        M3_TABLES / "price_drivers" / "price_driver_manifest.json",
    ),
    "seller_type": M3_TABLES / "seller_type_comparison" / "seller_type_comparison_summary.csv",
    "seller_manifest": first_existing_path(
        M3_QA / "seller_type_comparison" / "seller_type_manifest.json",
        M3_QA / "seller_type_comparison" / "seller_type_comparison_manifest.json",
        M3_TABLES / "seller_type_comparison" / "seller_type_comparison_manifest.json",
    ),
    "text_signals": M3_TABLES / "text_analysis" / "text_signal_summary.csv",
    "text_monthly": M3_TABLES / "text_analysis" / "text_keyword_monthly_frequency.csv",
    "text_manifest": first_existing_path(
        M3_QA / "text_analysis" / "text_signal_manifest.json",
        M3_TABLES / "text_analysis" / "text_signal_manifest.json",
    ),
    "segment_profile": M3_TABLES / "market_segmentation" / "segment_profile.csv",
    "segment_assignments": M3_MODELS / "market_segmentation" / "listing_segments.parquet",
    "analysis_ready_features": M3_MODELS / "price_drivers" / "analysis_ready_features.parquet",
    "segment_manifest": first_existing_path(
        M3_QA / "market_segmentation" / "segmentation_manifest.json",
        M3_TABLES / "market_segmentation" / "segmentation_manifest.json",
    ),
}

# Page-2 status inputs are presentation metadata, not analytical dependencies.
QUALITY_STATUS_SOURCES = {
    "m1_closeout": first_existing_path(
        OUTPUTS_DIR / "tables" / "milestone1_summary.json",
        OUTPUTS_DIR / "tables" / "milestone_1" / "milestone1_summary.json",
        OUTPUTS_DIR / "tables" / "milestone_1" / "silver_final" / "milestone1_summary.json",
        OUTPUTS_DIR / "qa" / "milestone_1" / "milestone1_summary.json",
    ),
    "m2_quality_gate": first_existing_path(
        OUTPUTS_DIR / "tables" / "milestone_2" / "quality_gate" / "milestone2_quality_gate_manifest.json",
        OUTPUTS_DIR / "tables" / "milestone_2" / "quality_gate" / "quality_gate_summary.csv",
        OUTPUTS_DIR / "qa" / "milestone_2" / "quality_gate" / "milestone2_quality_gate_manifest.json",
        OUTPUTS_DIR / "qa" / "milestone_2" / "milestone2_quality_gate_manifest.json",
        OUTPUTS_DIR / "tables" / "milestone_2" / "milestone2_quality_gate_manifest.json",
    ),
}



def show_progress(percent: int, label: str, *, final: bool = False) -> None:
    pct = max(0, min(100, int(percent)))
    filled = int(PROGRESS_WIDTH * pct / 100)
    bar = "#" * filled + "-" * (PROGRESS_WIDTH - filled)
    print(
        f"\rM4 gold [{bar}] {pct:3d}% complete | {100-pct:3d}% remaining | {label[:42]:42s}",
        end="\n" if final else "",
        flush=True,
    )


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


MANIFEST_READY_KEYS: dict[str, tuple[str, ...]] = {
    "market_map": ("m3_03_ready",),
    "temperature": ("m3_06_ready",),
    "price_drivers": ("m3_08_ready",),
    "seller": ("m3_09_ready",),
    "text": ("m3_11_ready",),
    "segments": ("ready",),
}


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _manifest_status_dict(payload: dict[str, Any]) -> dict[str, Any]:
    value = payload.get("status")
    return value if isinstance(value, dict) else {}


def _manifest_critical_failures(payload: dict[str, Any]) -> int:
    status = _manifest_status_dict(payload)
    value = status.get("critical_failures", payload.get("critical_failures", 0))
    return _safe_int(value, default=0)


def _manifest_overall_status(payload: dict[str, Any]) -> str:
    status = payload.get("status")
    if isinstance(status, dict):
        value = status.get("overall_status", payload.get("overall_status", ""))
    elif isinstance(status, str):
        value = payload.get("overall_status", status)
    else:
        value = payload.get("overall_status", "")
    return str(value or "").strip().upper()


def _manifest_ready_flag(payload: dict[str, Any], dependency_name: str) -> bool:
    if dependency_name == "monthly":
        if "m3_04_ready" in payload or "m3_05_ready" in payload:
            return bool(payload.get("m3_04_ready")) and bool(payload.get("m3_05_ready"))
        if "section_22_ready" in payload:
            return bool(payload.get("section_22_ready"))

    for key in MANIFEST_READY_KEYS.get(dependency_name, ()):
        if key in payload:
            return bool(payload.get(key))

    status = _manifest_status_dict(payload)
    if "ready" in status:
        return bool(status.get("ready"))
    if "ready" in payload:
        return bool(payload.get("ready"))
    return _manifest_overall_status(payload) == "PASS"


def _manifest_ready(payload: dict[str, Any], dependency_name: str) -> bool:
    if _manifest_critical_failures(payload) != 0:
        return False
    if _manifest_overall_status(payload) in {"FAIL", "FAILED", "CRITICAL"}:
        return False
    return _manifest_ready_flag(payload, dependency_name)


def _manifest_readiness_summary(payload: dict[str, Any], dependency_name: str) -> dict[str, Any]:
    return {
        "dependency": dependency_name,
        "ready": _manifest_ready_flag(payload, dependency_name),
        "critical_failures": _manifest_critical_failures(payload),
        "overall_status": _manifest_overall_status(payload) or None,
    }


def _require_columns(frame: pl.DataFrame, required: Iterable[str], source_name: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{source_name} is missing required columns: {missing}")


def _read_csv(path: Path) -> pl.DataFrame:
    return pl.read_csv(path, infer_schema_length=10000, try_parse_dates=False)


def _clean_text_expr(column: str) -> pl.Expr:
    value = pl.col(column).cast(pl.String, strict=False).str.strip_chars()
    return pl.when(value.is_null() | (value == "")).then(pl.lit(None, dtype=pl.String)).otherwise(value)


def _coalesced_text_expr(frame: pl.DataFrame, candidates: list[str], *, label: str) -> pl.Expr:
    available = [column for column in candidates if column in frame.columns]
    if not available:
        raise ValueError(f"No source column is available for {label}: candidates={candidates}")
    return pl.coalesce([_clean_text_expr(column) for column in available])


def normalize_neighborhood_source(source: pl.DataFrame) -> pl.DataFrame:
    """Return the dashboard-eligible city x neighborhood slice of the frozen M3 map summary.

    M3 map releases may expose canonical slug columns directly or retain the older
    city/neighborhood aliases. Gold accepts either naming shape, excludes rows with no
    neighborhood entity, removes exact duplicate rows only, and refuses conflicting
    duplicate city-neighborhood keys. No price/reliability metric is recomputed here.
    """
    normalized = source.with_columns(
        [
            _coalesced_text_expr(source, ["city_slug", "city"], label="city slug").alias("city_slug"),
            _coalesced_text_expr(
                source, ["neighborhood_slug", "neighborhood"], label="neighborhood slug"
            ).alias("neighborhood_slug"),
        ]
    )
    normalized = normalized.filter(
        pl.col("city_slug").is_not_null() & pl.col("neighborhood_slug").is_not_null()
    ).unique(maintain_order=True)
    duplicate_groups = (
        normalized.group_by(["city_slug", "neighborhood_slug"])
        .len()
        .filter(pl.col("len") > 1)
    )
    if duplicate_groups.height:
        preview = duplicate_groups.head(10).to_dicts()
        raise ValueError(
            "Frozen M3 neighborhood summary has conflicting duplicate city-neighborhood keys "
            f"after exact-row deduplication: groups={duplicate_groups.height}; preview={preview}"
        )
    if normalized.height == 0:
        raise ValueError("Frozen M3 neighborhood summary has no usable city-neighborhood rows.")
    return normalized


def _humanize(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return " ".join(part.capitalize() for part in re.split(r"[-_\s]+", text) if part)


def _normalize_month(frame: pl.DataFrame, column: str = "analysis_month") -> pl.DataFrame:
    return frame.with_columns(
        pl.col(column).cast(pl.String).str.slice(0, 7).alias(column)
    ).with_columns(
        pl.col(column).str.replace_all("-", "").cast(pl.Int32, strict=False).alias("month_key")
    )


def _normalized_location_grain(frame: pl.DataFrame, source_name: str) -> pl.DataFrame:
    """Keep only rows that can map unambiguously to the conformed location dimension.

    Frozen monthly sources may contain a null-city aggregate bucket produced by grouping
    listings whose city is missing. Gold does not relabel that bucket as a real city or
    national row. National aggregates remain available separately, while unmapped
    location buckets are excluded from location-grained facts.
    """
    _require_columns(frame, ["entity_level", "city_slug", "neighborhood_slug"], source_name)
    normalized = frame.with_columns(
        [
            _clean_text_expr("entity_level").str.to_lowercase().alias("entity_level"),
            _clean_text_expr("city_slug").alias("city_slug"),
            _clean_text_expr("neighborhood_slug").alias("neighborhood_slug"),
        ]
    )
    valid = (
        (
            (pl.col("entity_level") == "national")
            & pl.col("city_slug").is_null()
            & pl.col("neighborhood_slug").is_null()
        )
        | (
            (pl.col("entity_level") == "city")
            & pl.col("city_slug").is_not_null()
            & pl.col("neighborhood_slug").is_null()
        )
        | (
            (pl.col("entity_level") == "neighborhood")
            & pl.col("city_slug").is_not_null()
            & pl.col("neighborhood_slug").is_not_null()
        )
    )
    return normalized.filter(valid)


def _invalid_location_grain_count(frame: pl.DataFrame, source_name: str) -> int:
    normalized = frame.with_columns(
        [
            _clean_text_expr("entity_level").str.to_lowercase().alias("entity_level"),
            _clean_text_expr("city_slug").alias("city_slug"),
            _clean_text_expr("neighborhood_slug").alias("neighborhood_slug"),
        ]
    )
    return int(normalized.height - _normalized_location_grain(normalized, source_name).height)


def _with_location_key(frame: pl.DataFrame) -> pl.DataFrame:
    _require_columns(frame, ["entity_level", "city_slug", "neighborhood_slug"], "location-grain frame")
    city = pl.col("city_slug").cast(pl.String)
    neighborhood = pl.col("neighborhood_slug").cast(pl.String)
    return frame.with_columns(
        pl.when(pl.col("entity_level") == "national")
        .then(pl.lit("NATIONAL"))
        .when(pl.col("entity_level") == "city")
        .then(pl.concat_str([pl.lit("CITY:"), city]))
        .otherwise(pl.concat_str([pl.lit("NEIGHBORHOOD:"), city, pl.lit(":"), neighborhood]))
        .alias("location_key")
    )


def _write_parquet(frame: pl.DataFrame, path: Path) -> None:
    atomic_sink_parquet(frame.lazy(), path)


def _source_version(payload: dict[str, Any], fallback: str = "unknown") -> str:
    preferred_keys = (
        "version",
        "analysis_version",
        "market_map_version",
        "monthly_market_version",
        "market_temperature_version",
        "price_driver_version",
        "seller_comparison_version",
        "text_signal_version",
        "segmentation_version",
    )
    for key in preferred_keys:
        value = payload.get(key)
        if isinstance(value, (str, int, float)) and not isinstance(value, bool) and str(value).strip():
            return str(value)
    for key, value in payload.items():
        if (
            str(key).endswith("_version")
            and isinstance(value, (str, int, float))
            and not isinstance(value, bool)
            and str(value).strip()
        ):
            return str(value)
    return fallback


def validate_upstream() -> tuple[dict[str, dict[str, Any]], list[Check]]:
    required_source_keys = (
        "monthly_market", "monthly_supply", "monthly_manifest",
        "neighborhood_market", "market_map_manifest",
        "market_temperature", "market_temperature_manifest",
        "price_driver_effects", "price_driver_diagnostics", "price_driver_benchmark",
        "price_driver_importance", "avm_error_analysis", "price_driver_manifest",
        "seller_type", "seller_manifest",
        "text_signals", "text_monthly", "text_manifest",
        "segment_profile", "segment_assignments", "analysis_ready_features", "segment_manifest",
    )
    missing = [name for name in required_source_keys if not SOURCES[name].exists()]
    if missing:
        details = ", ".join(f"{name}={relative_to_project(SOURCES[name])}" for name in missing)
        raise FileNotFoundError(f"Required frozen M3 artifacts are missing: {details}")

    manifests = {
        "monthly": _load_json(SOURCES["monthly_manifest"]),
        "market_map": _load_json(SOURCES["market_map_manifest"]),
        "temperature": _load_json(SOURCES["market_temperature_manifest"]),
        "price_drivers": _load_json(SOURCES["price_driver_manifest"]),
        "seller": _load_json(SOURCES["seller_manifest"]),
        "text": _load_json(SOURCES["text_manifest"]),
        "segments": _load_json(SOURCES["segment_manifest"]),
    }
    checks: list[Check] = []
    for name, payload in manifests.items():
        checks.append(
            make_check(
                f"upstream_{name}_ready", "dependency",
                _manifest_readiness_summary(payload, name),
                "accepted ready flag and critical_failures=0",
                _manifest_ready(payload, name),
                critical=True,
                notes="Gold consumes accepted canonical M3 outputs and never repairs an upstream failure.",
            )
        )
    temperature_frozen = bool(manifests["temperature"].get("freeze_authorized", False))
    checks.append(
        make_check(
            "market_temperature_frozen", "dependency", temperature_frozen, True,
            temperature_frozen, critical=True,
            notes="Canonical all-city Market Temperature must be frozen before Gold publication.",
        )
    )
    failed = [
        check.check_id for check in checks
        if str(check.status).upper() in {"FAIL", "CRITICAL"} and check.critical
    ]
    if failed:
        raise RuntimeError("Frozen M3 dependency validation failed: " + ", ".join(failed))
    return manifests, checks

def build_market_monthly(monthly_version: str) -> pl.DataFrame:
    market = _read_csv(SOURCES["monthly_market"])
    supply = _read_csv(SOURCES["monthly_supply"])
    _require_columns(
        market,
        [
            "analysis_month",
            "entity_level",
            "city_slug",
            "neighborhood_slug",
            "raw_listing_count",
            "deduplicated_listing_count",
            "deduplicated_supply_mom_pct",
            "price_listing_n",
            "median_asking_price_per_sqm_toman",
            "median_price_mom_pct",
            "price_reliable_flag",
            "market_scope",
            "analysis_version",
        ],
        "monthly_market_summary.csv",
    )
    _require_columns(
        supply,
        [
            "analysis_month",
            "entity_level",
            "city_slug",
            "neighborhood_slug",
            "market_scope",
            "raw_listing_count",
            "deduplicated_listing_count",
            "deduplicated_supply_mom_pct",
            "duplicate_supply_reduction_rate",
        ],
        "monthly_supply_summary.csv",
    )

    market = _normalized_location_grain(market, "monthly_market_summary.csv")
    supply = _normalized_location_grain(supply, "monthly_supply_summary.csv")

    market = _normalize_month(market)
    market = _with_location_key(market).with_columns(
        [
            pl.lit("price_and_supply").alias("series_kind"),
            pl.lit("apartment").alias("property_scope"),
            pl.lit("family:apartment").alias("property_key"),
            pl.lit("sale").alias("price_regime"),
            pl.lit("regime:sale").alias("price_regime_key"),
        ]
    )

    extra_supply = supply.filter(pl.col("market_scope").is_in(["residential_sale", "all_market"]))
    extra_supply = _normalize_month(extra_supply)
    extra_supply = _with_location_key(extra_supply).with_columns(
        [
            pl.lit("supply_only").alias("series_kind"),
            pl.when(pl.col("market_scope") == "residential_sale")
            .then(pl.lit("residential"))
            .otherwise(pl.lit("all"))
            .alias("property_scope"),
            pl.lit(None, dtype=pl.String).alias("property_key"),
            pl.when(pl.col("market_scope") == "residential_sale")
            .then(pl.lit("sale"))
            .otherwise(pl.lit(None, dtype=pl.String))
            .alias("price_regime"),
            pl.when(pl.col("market_scope") == "residential_sale")
            .then(pl.lit("regime:sale"))
            .otherwise(pl.lit(None, dtype=pl.String))
            .alias("price_regime_key"),
            pl.lit(None, dtype=pl.Int64).alias("price_listing_n"),
            pl.lit(None, dtype=pl.Float64).alias("median_asking_price_per_sqm_toman"),
            pl.lit(None, dtype=pl.Float64).alias("p25_asking_price_per_sqm_toman"),
            pl.lit(None, dtype=pl.Float64).alias("p75_asking_price_per_sqm_toman"),
            pl.lit(None, dtype=pl.Float64).alias("median_price_mom_pct"),
            pl.lit(None, dtype=pl.Boolean).alias("price_reliable_flag"),
            pl.lit(None, dtype=pl.String).alias("supply_population"),
            pl.lit(None, dtype=pl.String).alias("price_population"),
            pl.lit(monthly_version).alias("analysis_version"),
        ]
    )

    columns = [
        "analysis_month",
        "month_key",
        "entity_level",
        "location_key",
        "city_slug",
        "neighborhood_slug",
        "market_scope",
        "series_kind",
        "property_scope",
        "property_key",
        "price_regime",
        "price_regime_key",
        "raw_listing_count",
        "deduplicated_listing_count",
        "deduplicated_supply_mom_pct",
        "raw_supply_mom_pct",
        "duplicate_supply_reduction_rate",
        "price_listing_n",
        "median_asking_price_per_sqm_toman",
        "p25_asking_price_per_sqm_toman",
        "p75_asking_price_per_sqm_toman",
        "median_price_mom_pct",
        "price_reliable_flag",
        "supply_population",
        "price_population",
        "analysis_version",
    ]
    for name in columns:
        if name not in market.columns:
            market = market.with_columns(pl.lit(None).alias(name))
        if name not in extra_supply.columns:
            extra_supply = extra_supply.with_columns(pl.lit(None).alias(name))
    return pl.concat([market.select(columns), extra_supply.select(columns)], how="diagonal_relaxed").sort(
        ["market_scope", "entity_level", "city_slug", "neighborhood_slug", "analysis_month"]
    )


def _market_map_minimum_n(manifest: dict[str, Any], default: int = 30) -> int:
    rules = manifest.get("rules") if isinstance(manifest.get("rules"), dict) else {}
    candidates = (
        rules.get("minimum_sample_size"),
        manifest.get("minimum_valid_listings"),
        manifest.get("minimum_sample_size"),
    )
    for value in candidates:
        parsed = _safe_int(value, default=-1)
        if parsed > 0:
            return parsed
    return default


def build_neighborhood_market(manifest: dict[str, Any]) -> pl.DataFrame:
    source = normalize_neighborhood_source(_read_csv(SOURCES["neighborhood_market"]))
    _require_columns(
        source,
        [
            "city_slug",
            "neighborhood_slug",
            "listing_n",
            "median_asking_price_per_sqm_toman",
            "p25_asking_price_per_sqm_toman",
            "p75_asking_price_per_sqm_toman",
            "iqr_asking_price_per_sqm_toman",
            "reliable_flag",
            "reliability_status",
            "reliability_reason",
            "metric_definition",
            "analysis_version",
        ],
        "neighborhood_market_summary.csv",
    )
    min_n = _market_map_minimum_n(manifest)
    base = source.with_columns(
        [
            pl.concat_str(
                [
                    pl.lit("NEIGHBORHOOD:"),
                    pl.col("city_slug").cast(pl.String),
                    pl.lit(":"),
                    pl.col("neighborhood_slug").cast(pl.String),
                ]
            ).alias("location_key"),
            pl.lit("apartment").alias("property_family"),
            pl.lit("family:apartment").alias("property_key"),
            pl.lit("sale").alias("price_regime"),
            pl.lit("regime:sale").alias("price_regime_key"),
            pl.col("listing_n").alias("valid_price_n"),
            pl.lit(min_n).alias("minimum_n_threshold"),
        ]
    )
    # Static canonical ranks are presentation metadata over the already-published
    # reliable price-map rows. They do not refit or alter an upstream estimate.
    ranking = (
        base.filter(pl.col("reliable_flag") == True)
        .with_columns(
            [
                pl.col("median_asking_price_per_sqm_toman")
                .rank(method="dense")
                .cast(pl.Int64)
                .alias("affordable_rank"),
                (-pl.col("median_asking_price_per_sqm_toman"))
                .rank(method="dense")
                .cast(pl.Int64)
                .alias("expensive_rank"),
            ]
        )
        .select(["location_key", "affordable_rank", "expensive_rank"])
    )
    return (
        base.join(ranking, on="location_key", how="left")
        .select(
            [
                "location_key",
                "city_slug",
                "neighborhood_slug",
                "property_family",
                "property_key",
                "price_regime",
                "price_regime_key",
                "listing_n",
                "valid_price_n",
                "median_asking_price_per_sqm_toman",
                "p25_asking_price_per_sqm_toman",
                "p75_asking_price_per_sqm_toman",
                "iqr_asking_price_per_sqm_toman",
                "reliable_flag",
                "affordable_rank",
                "expensive_rank",
                "reliability_status",
                "reliability_reason",
                "minimum_n_threshold",
                "metric_definition",
                "analysis_version",
            ]
        )
        .sort(["city_slug", "median_asking_price_per_sqm_toman"], descending=[False, True])
    )


def build_market_temperature(manifest: dict[str, Any]) -> pl.DataFrame:
    source = _read_csv(SOURCES["market_temperature"])
    _require_columns(
        source,
        [
            "entity_level", "city_slug", "neighborhood_slug",
            "price_trend_pct_per_month", "supply_trend_pct_per_month",
            "price_direction_consistency", "supply_direction_consistency", "trend_stability_score",
            "price_month_count", "supply_month_count", "usable_month_count", "sample_size",
            "median_monthly_listing_count", "evidence_multiplier", "price_rank", "supply_signal_rank",
            "market_temperature_score", "market_temperature_percentile", "market_temperature_label",
            "market_temperature_rank", "proxy_definition", "supply_population", "price_population",
            "listing_activity_interpretation", "analysis_version",
        ],
        "market_temperature_summary.csv",
    )
    gate = manifest.get("definition", {}).get("professor_facing_reliability_gate", {})
    minimum_months = int(gate.get("minimum_reliable_price_months", 5))
    minimum_activity_months = int(gate.get("minimum_listing_activity_months", minimum_months))
    minimum_n = int(gate.get("minimum_sample_size", 100))
    source = _with_location_key(source)
    reliable = (
        (pl.col("price_month_count").cast(pl.Int64, strict=False) >= minimum_months)
        & (pl.col("supply_month_count").cast(pl.Int64, strict=False) >= minimum_activity_months)
        & (pl.col("sample_size").cast(pl.Int64, strict=False) >= minimum_n)
    )
    return (
        source.with_columns(
            [
                pl.col("supply_trend_pct_per_month").alias("listing_activity_trend_pct_per_month"),
                pl.col("supply_direction_consistency").alias("listing_activity_direction_consistency"),
                pl.col("supply_month_count").alias("listing_activity_month_count"),
                pl.col("supply_signal_rank").alias("listing_activity_signal_rank"),
                pl.col("supply_population").alias("listing_activity_population"),
                reliable.alias("temperature_reliability_eligible_flag"),
                # Kept for the later catalog name, but intentionally all-city. §28/four-city
                # presentation is controlled by dim_location.is_four_city, not by the estimate.
                (reliable & (pl.col("entity_level") == "neighborhood")).alias("professor_facing_eligible_flag"),
                pl.lit(minimum_months).alias("temperature_minimum_price_months"),
                pl.lit(minimum_activity_months).alias("temperature_minimum_activity_months"),
                pl.lit(minimum_n).alias("temperature_minimum_n"),
                pl.lit("apartment").alias("property_family"),
                pl.lit("family:apartment").alias("property_key"),
                pl.lit("sale").alias("price_regime"),
                pl.lit("regime:sale").alias("price_regime_key"),
                pl.col("sample_size").alias("temperature_sample_n"),
            ]
        )
        .select(
            [
                "entity_level", "location_key", "city_slug", "neighborhood_slug",
                "property_family", "property_key", "price_regime", "price_regime_key",
                "price_trend_pct_per_month", "listing_activity_trend_pct_per_month",
                "price_direction_consistency", "listing_activity_direction_consistency",
                "trend_stability_score", "price_month_count", "listing_activity_month_count",
                "usable_month_count", "temperature_sample_n", "median_monthly_listing_count",
                "evidence_multiplier", "price_rank", "listing_activity_signal_rank",
                "market_temperature_score", "market_temperature_percentile",
                "market_temperature_label", "market_temperature_rank",
                "temperature_reliability_eligible_flag", "professor_facing_eligible_flag",
                "temperature_minimum_price_months", "temperature_minimum_activity_months",
                "temperature_minimum_n", "proxy_definition", "listing_activity_population",
                "price_population", "listing_activity_interpretation", "analysis_version",
            ]
        )
        .sort(["entity_level", "market_temperature_rank"])
    )

def build_location_market(
    market_map_manifest: dict[str, Any], temperature_manifest: dict[str, Any]
) -> pl.DataFrame:
    """Merge price-map and temperature outputs at one location grain.

    Neighborhood price-map rows and city/neighborhood temperature rows describe the
    same location entity at different analytical scopes. Keeping them in one mart
    removes a redundant fact table without recomputing either estimate.
    """
    price = build_neighborhood_market(market_map_manifest).with_columns(
        [
            pl.lit("neighborhood").alias("entity_level"),
            pl.lit(True).alias("price_map_available_flag"),
            pl.col("reliability_status").alias("price_reliability_status"),
            pl.col("reliability_reason").alias("price_reliability_reason"),
            pl.col("analysis_version").alias("price_analysis_version"),
        ]
    ).drop(["reliability_status", "reliability_reason", "analysis_version"])
    temperature = build_market_temperature(temperature_manifest).with_columns(
        [
            pl.lit(True).alias("temperature_available_flag"),
            pl.col("analysis_version").alias("temperature_analysis_version"),
        ]
    ).drop("analysis_version")

    temp_metrics = [
        c for c in temperature.columns
        if c not in {
            "entity_level", "location_key", "city_slug", "neighborhood_slug",
            "property_family", "property_key", "price_regime", "price_regime_key",
        }
    ]
    temp_neighborhood = temperature.filter(pl.col("entity_level") == "neighborhood").select(
        ["location_key", *temp_metrics]
    )
    merged = price.join(temp_neighborhood, on="location_key", how="left")
    merged = merged.with_columns(
        pl.col("temperature_available_flag").fill_null(False).alias("temperature_available_flag")
    )

    existing = price.select("location_key")
    temp_only_neighborhood = (
        temperature.filter(pl.col("entity_level") == "neighborhood")
        .join(existing, on="location_key", how="anti")
        .with_columns(pl.lit(False).alias("price_map_available_flag"))
    )
    temp_city = temperature.filter(pl.col("entity_level") == "city").with_columns(
        pl.lit(False).alias("price_map_available_flag")
    )

    out = pl.concat([merged, temp_only_neighborhood, temp_city], how="diagonal_relaxed")
    out = out.with_columns(
        pl.concat_str(
            [
                pl.col("price_analysis_version").cast(pl.String, strict=False).fill_null("no_price_map"),
                pl.lit(" | "),
                pl.col("temperature_analysis_version").cast(pl.String, strict=False).fill_null("no_temperature"),
            ]
        ).alias("method_version")
    )
    # One location key must correspond to one entity row.
    duplicates = out.select(["entity_level", "location_key"]).is_duplicated().sum()
    if int(duplicates) != 0:
        raise ValueError(f"mart_location_market would contain {int(duplicates)} duplicate entity/location rows")
    return out.sort(["entity_level", "city_slug", "neighborhood_slug"], nulls_last=True)


def _driver_feature_metadata() -> pl.DataFrame:
    rows = [
        ("primary_area_sqm", "Primary area", "structural"),
        ("rooms_count_num", "Rooms", "structural"),
        ("building_age_years", "Building age", "structural"),
        ("floor_num", "Floor", "structural"),
        ("has_elevator_bool", "Elevator", "amenity"),
        ("has_parking_bool", "Parking", "amenity"),
        ("has_warehouse_bool", "Storage", "amenity"),
        ("has_balcony_bool", "Balcony", "amenity"),
        ("is_rebuilt_bool", "Rebuilt", "structural"),
        ("construction_year_before_1370_flag", "Built before 1370", "structural"),
    ]
    return pl.DataFrame(rows, schema=["base_feature", "base_display_name", "feature_group"], orient="row")


def build_price_driver_effects(version: str) -> pl.DataFrame:
    source = _read_csv(SOURCES["price_driver_effects"])
    _require_columns(
        source,
        [
            "feature",
            "base_feature",
            "contrast_definition",
            "adjusted_effect_pct",
            "adjusted_effect_ci_low",
            "adjusted_effect_ci_high",
            "uncertainty_method",
            "interpretation",
            "reference_n",
        ],
        "price_driver_summary.csv",
    )
    meta = _driver_feature_metadata()
    out = source.join(meta, on="base_feature", how="left")
    return (
        out.with_columns(
            [
                pl.col("feature").alias("feature_id"),
                pl.when(pl.col("base_display_name").is_not_null())
                .then(
                    pl.when(pl.col("feature") == pl.col("base_feature"))
                    .then(pl.col("base_display_name"))
                    .otherwise(
                        pl.concat_str(
                            [
                                pl.col("base_display_name"),
                                pl.lit(": "),
                                pl.col("feature").cast(pl.String).str.split_exact(":", 1).struct.field("field_1"),
                            ]
                        )
                    )
                )
                .otherwise(pl.col("feature").cast(pl.String).str.replace_all("_", " "))
                .alias("feature_display_name"),
                pl.coalesce([pl.col("feature_group"), pl.lit("property_characteristic")]).alias("feature_group"),
                pl.col("adjusted_effect_ci_low").alias("ci_lower_pct"),
                pl.col("adjusted_effect_ci_high").alias("ci_upper_pct"),
                pl.col("reference_n").alias("sample_n"),
                pl.lit("all_sales").alias("population"),
                pl.lit("ridge").alias("model_name"),
                pl.lit("test").alias("evaluation_split"),
                pl.lit(version).alias("method_version"),
            ]
        )
        .select(
            [
                "feature_id",
                "feature_display_name",
                "base_feature",
                "feature_group",
                "contrast_definition",
                "adjusted_effect_pct",
                "ci_lower_pct",
                "ci_upper_pct",
                "uncertainty_method",
                "sample_n",
                "population",
                "model_name",
                "evaluation_split",
                "interpretation",
                "method_version",
            ]
        )
        .sort("feature_id")
    )


def build_price_driver_importance(version: str) -> pl.DataFrame:
    source = _read_csv(SOURCES["price_driver_importance"])
    _require_columns(
        source,
        [
            "feature_block",
            "columns",
            "heldout_rmse_log_increase_mean",
            "heldout_rmse_log_increase_sd",
            "test_sample_n",
            "interpretation",
        ],
        "price_driver_permutation_importance.csv",
    )
    name_map = {
        "location": "Location controls (City + Neighborhood; grouped)",
        "property_type": "Property-type controls (Family + Category; grouped)",
        "time": "Time control (Month)",
        "primary_area_sqm": "Primary area",
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
    group_map = {
        "location": "location_control",
        "property_type": "property_control",
        "time": "time_control",
        "has_elevator_bool": "amenity",
        "has_parking_bool": "amenity",
        "has_warehouse_bool": "amenity",
        "has_balcony_bool": "amenity",
    }
    rows = source.to_dicts()
    for row in rows:
        block = str(row["feature_block"])
        row["feature_or_block_id"] = block
        row["feature_or_block_name"] = name_map.get(block, _humanize(block) or block)
        row["feature_group"] = group_map.get(block, "structural")
        row["importance_type"] = (
            "held_out_grouped_permutation_rmse_log_increase"
            if ";" in str(row.get("columns") or "")
            else "held_out_permutation_rmse_log_increase"
        )
        row["model_name"] = "ridge"
        row["population"] = "all_sales"
        row["evaluation_split"] = "test"
        row["method_version"] = version
    return (
        pl.DataFrame(rows)
        .rename(
            {
                "heldout_rmse_log_increase_mean": "permutation_importance",
                "heldout_rmse_log_increase_sd": "permutation_importance_sd",
                "test_sample_n": "sample_n",
            }
        )
        .select(
            [
                "feature_or_block_id",
                "feature_or_block_name",
                "feature_group",
                "columns",
                "permutation_importance",
                "permutation_importance_sd",
                "importance_type",
                "model_name",
                "population",
                "evaluation_split",
                "sample_n",
                "interpretation",
                "method_version",
            ]
        )
        .sort("permutation_importance", descending=True)
    )


def _model_quality_columns() -> list[str]:
    return [
        "record_type",
        "model_name",
        "population",
        "evaluation_split",
        "evaluation_stage",
        "is_primary_model",
        "n",
        "rmse_log",
        "r2_log",
        "median_ape_pct",
        "p75_ape_pct",
        "p90_ape_pct",
        "within_20_pct",
        "within_30_pct",
        "within_50_pct",
        "median_absolute_error_psm_toman",
        "mae_psm_toman",
        "error_scope",
        "city_slug",
        "property_family",
        "reliability_status",
        "benchmark_role",
        "method_version",
    ]


def build_model_quality(version: str) -> pl.DataFrame:
    diagnostics = _read_csv(SOURCES["price_driver_diagnostics"])
    benchmark = _read_csv(SOURCES["price_driver_benchmark"])
    errors = _read_csv(SOURCES["avm_error_analysis"])
    metric_cols = [
        "n",
        "rmse_log",
        "r2_log",
        "median_absolute_percentage_error_pct",
        "p75_absolute_percentage_error_pct",
        "p90_absolute_percentage_error_pct",
        "within_20pct_share_pct",
        "within_30pct_share_pct",
        "within_50pct_share_pct",
        "median_abs_error_price_per_sqm_toman",
        "mae_price_per_sqm_toman",
    ]
    _require_columns(diagnostics, ["model", "split", "evaluation_stage", *metric_cols], "price_driver_model_diagnostics.csv")
    _require_columns(benchmark, ["population", "model", "split", "benchmark_role", *metric_cols], "price_model_benchmark.csv")
    _require_columns(
        errors,
        [
            "n",
            "median_ape_pct",
            "p75_ape_pct",
            "p90_ape_pct",
            "within_20pct_share_pct",
            "within_30pct_share_pct",
            "within_50pct_share_pct",
            "median_abs_error_toman_per_sqm",
            "mae_toman_per_sqm",
            "error_scope",
            "reliability_status",
        ],
        "avm_error_analysis.csv",
    )

    diag = diagnostics.with_columns(
        [
            pl.lit("primary_diagnostics").alias("record_type"),
            pl.col("model").alias("model_name"),
            pl.lit("all_sales").alias("population"),
            pl.col("split").alias("evaluation_split"),
            (pl.col("model") == "ridge").alias("is_primary_model"),
            pl.col("median_absolute_percentage_error_pct").alias("median_ape_pct"),
            pl.col("p75_absolute_percentage_error_pct").alias("p75_ape_pct"),
            pl.col("p90_absolute_percentage_error_pct").alias("p90_ape_pct"),
            pl.col("within_20pct_share_pct").alias("within_20_pct"),
            pl.col("within_30pct_share_pct").alias("within_30_pct"),
            pl.col("within_50pct_share_pct").alias("within_50_pct"),
            pl.col("median_abs_error_price_per_sqm_toman").alias("median_absolute_error_psm_toman"),
            pl.col("mae_price_per_sqm_toman").alias("mae_psm_toman"),
            pl.lit(None, dtype=pl.String).alias("error_scope"),
            pl.lit(None, dtype=pl.String).alias("city_slug"),
            pl.lit(None, dtype=pl.String).alias("property_family"),
            pl.lit(None, dtype=pl.String).alias("reliability_status"),
            pl.lit(None, dtype=pl.String).alias("benchmark_role"),
            pl.lit(version).alias("method_version"),
        ]
    )

    bench = benchmark.with_columns(
        [
            pl.lit("fair_benchmark").alias("record_type"),
            pl.col("model").alias("model_name"),
            pl.col("split").alias("evaluation_split"),
            pl.lit("fair_same_rows_features_preprocessing").alias("evaluation_stage"),
            pl.lit(False).alias("is_primary_model"),
            pl.col("median_absolute_percentage_error_pct").alias("median_ape_pct"),
            pl.col("p75_absolute_percentage_error_pct").alias("p75_ape_pct"),
            pl.col("p90_absolute_percentage_error_pct").alias("p90_ape_pct"),
            pl.col("within_20pct_share_pct").alias("within_20_pct"),
            pl.col("within_30pct_share_pct").alias("within_30_pct"),
            pl.col("within_50pct_share_pct").alias("within_50_pct"),
            pl.col("median_abs_error_price_per_sqm_toman").alias("median_absolute_error_psm_toman"),
            pl.col("mae_price_per_sqm_toman").alias("mae_psm_toman"),
            pl.lit(None, dtype=pl.String).alias("error_scope"),
            pl.lit(None, dtype=pl.String).alias("city_slug"),
            pl.lit(None, dtype=pl.String).alias("property_family"),
            pl.lit(None, dtype=pl.String).alias("reliability_status"),
            pl.lit(version).alias("method_version"),
        ]
    )

    for optional in ["city_slug", "property_family"]:
        if optional not in errors.columns:
            errors = errors.with_columns(pl.lit(None, dtype=pl.String).alias(optional))
    avm = errors.with_columns(
        [
            pl.lit("avm_error_segment").alias("record_type"),
            pl.lit("ridge").alias("model_name"),
            pl.lit("all_sales").alias("population"),
            pl.lit("test").alias("evaluation_split"),
            pl.lit("heldout_error_analysis").alias("evaluation_stage"),
            pl.lit(True).alias("is_primary_model"),
            pl.lit(None, dtype=pl.Float64).alias("rmse_log"),
            pl.lit(None, dtype=pl.Float64).alias("r2_log"),
            pl.col("within_20pct_share_pct").alias("within_20_pct"),
            pl.col("within_30pct_share_pct").alias("within_30_pct"),
            pl.col("within_50pct_share_pct").alias("within_50_pct"),
            pl.col("median_abs_error_toman_per_sqm").alias("median_absolute_error_psm_toman"),
            pl.col("mae_toman_per_sqm").alias("mae_psm_toman"),
            pl.lit(None, dtype=pl.String).alias("benchmark_role"),
            pl.lit(version).alias("method_version"),
        ]
    )

    columns = _model_quality_columns()
    return pl.concat(
        [diag.select(columns), bench.select(columns), avm.select(columns)],
        how="diagonal_relaxed",
    )


def build_seller_type(version: str) -> pl.DataFrame:
    source = _read_csv(SOURCES["seller_type"])
    _require_columns(source, ["comparison", "agency_label", "personal_label"], "seller_type_comparison_summary.csv")

    def coalesce(candidates: list[str], dtype: pl.DataType) -> pl.Expr:
        available = [name for name in candidates if name in source.columns]
        if not available:
            return pl.lit(None, dtype=dtype)
        return pl.coalesce([pl.col(name).cast(dtype, strict=False) for name in available])

    adapted = source.with_columns(
        [
            coalesce(["agency_n", "agency_sample_size"], pl.Int64).alias("agency_n"),
            coalesce(["personal_n", "personal_sample_size"], pl.Int64).alias("personal_n"),
            coalesce(["agency_median_asking_psm_toman", "agency_median_price_per_sqm"], pl.Float64)
            .alias("agency_median_asking_psm_toman"),
            coalesce(["personal_median_asking_psm_toman", "personal_median_price_per_sqm"], pl.Float64)
            .alias("personal_median_asking_psm_toman"),
            coalesce(["raw_median_difference_pct", "raw_price_difference_pct"], pl.Float64)
            .alias("raw_median_difference_pct"),
            coalesce(["raw_mann_whitney_p_value", "raw_p_value", "mann_whitney_p_value"], pl.Float64)
            .alias("raw_mann_whitney_p_value"),
            coalesce(["adjusted_crossfit_difference_pct", "adjusted_price_difference_pct"], pl.Float64)
            .alias("adjusted_crossfit_difference_pct"),
            coalesce(["adjusted_ci_low_pct"], pl.Float64).alias("adjusted_ci_low_pct"),
            coalesce(["adjusted_ci_high_pct"], pl.Float64).alias("adjusted_ci_high_pct"),
            coalesce(["adjusted_welch_p_value", "adjusted_p_value"], pl.Float64).alias("adjusted_welch_p_value"),
            coalesce(["stratified_difference_pct", "stratified_price_difference_pct"], pl.Float64)
            .alias("stratified_difference_pct"),
            coalesce(["crossfit_rows", "total_sample_size"], pl.Int64).alias("crossfit_rows"),
            coalesce(["interpretation", "interpretation_note"], pl.String).alias("interpretation"),
            coalesce(["analysis_version"], pl.String).fill_null(version).alias("analysis_version"),
        ]
    )
    required_after = [
        "agency_n", "personal_n", "adjusted_crossfit_difference_pct",
        "adjusted_ci_low_pct", "adjusted_ci_high_pct",
    ]
    if any(adapted.get_column(column).null_count() == adapted.height for column in required_after):
        missing = [column for column in required_after if adapted.get_column(column).null_count() == adapted.height]
        raise ValueError(f"seller_type_comparison_summary.csv cannot supply canonical fields: {missing}")
    total = pl.col("agency_n") + pl.col("personal_n")
    return (
        adapted.with_columns(
            [
                total.alias("sample_size"),
                (pl.col("agency_n") / total * 100.0).alias("agency_listing_share_pct"),
                (pl.col("personal_n") / total * 100.0).alias("personal_listing_share_pct"),
                pl.when(pl.col("adjusted_ci_low_pct") > 0).then(pl.lit("positive"))
                .when(pl.col("adjusted_ci_high_pct") < 0).then(pl.lit("negative"))
                .otherwise(pl.lit("not_clear")).alias("seller_statistical_status"),
                pl.lit("sale_seller_valid").alias("population"),
                pl.lit("sale").alias("price_regime"),
                pl.lit("regime:sale").alias("price_regime_key"),
                pl.lit(version).alias("method_version"),
            ]
        )
        .select(
            [
                "comparison", "agency_label", "personal_label",
                "agency_n", "personal_n", "sample_size",
                "agency_listing_share_pct", "personal_listing_share_pct",
                "agency_median_asking_psm_toman", "personal_median_asking_psm_toman",
                "raw_median_difference_pct", "raw_mann_whitney_p_value",
                "adjusted_crossfit_difference_pct", "adjusted_ci_low_pct",
                "adjusted_ci_high_pct", "adjusted_welch_p_value",
                "stratified_difference_pct", "crossfit_rows",
                "seller_statistical_status", "population", "price_regime",
                "price_regime_key", "interpretation", "method_version",
            ]
        )
    )

def build_text_signals(version: str) -> pl.DataFrame:
    source = _read_csv(SOURCES["text_signals"])
    _require_columns(
        source,
        [
            "keyword",
            "keyword_fa",
            "manual_precision",
            "precision_validation_status",
            "analysis_status",
            "heldout_test_n",
            "positive_n",
            "negative_n",
            "adjusted_residual_difference_pct",
            "adjusted_ci_low_pct",
            "adjusted_ci_high_pct",
            "adjusted_welch_p_value",
            "adjusted_fdr_bh_q_value",
            "adjusted_fdr_bh_status",
            "interpretation",
            "analysis_version",
        ],
        "text_signal_summary.csv",
    )
    filtered = source.filter(pl.col("keyword").is_in(sorted(EXPECTED_TEXT_SIGNALS)))
    return filtered.with_columns(
        [
            pl.col("keyword").alias("keyword_family"),
            pl.col("keyword_fa").alias("keyword_display_name"),
            pl.col("precision_validation_status").alias("precision_status"),
            pl.col("positive_n").alias("signal_n"),
            pl.col("heldout_test_n").alias("sample_n"),
            pl.col("adjusted_residual_difference_pct").alias("adjusted_effect_pct"),
            pl.col("adjusted_welch_p_value").alias("p_value"),
            pl.col("adjusted_fdr_bh_q_value").alias("q_value"),
            (pl.col("adjusted_fdr_bh_status") == "SIGNIFICANT_FDR_0_05").alias("fdr_significant_flag"),
            pl.lit("sale").alias("price_regime"),
            pl.lit("regime:sale").alias("price_regime_key"),
            pl.lit(version).alias("method_version"),
        ]
    ).select(
        [
            "keyword_family",
            "keyword_display_name",
            "manual_precision",
            "precision_status",
            "analysis_status",
            "signal_n",
            "negative_n",
            "sample_n",
            "adjusted_effect_pct",
            "adjusted_ci_low_pct",
            "adjusted_ci_high_pct",
            "p_value",
            "q_value",
            "fdr_significant_flag",
            "adjusted_fdr_bh_status",
            "price_regime",
            "price_regime_key",
            "interpretation",
            "method_version",
        ]
    )


def build_text_monthly(version: str) -> pl.DataFrame:
    """Adapt the accepted M3-11 Keyword x Month schema to the stable Gold schema.

    Canonical M3-11 emits ``positive_n``, ``population_n`` and ``positive_rate``.
    Gold keeps dashboard-facing names that describe the sale-valid population and
    matched keyword listings. This is a pure semantic rename; no metric is recomputed.
    """
    source = _read_csv(SOURCES["text_monthly"])
    _require_columns(
        source,
        ["keyword", "keyword_fa", "analysis_month", "positive_n", "population_n", "positive_rate"],
        "text_keyword_monthly_frequency.csv",
    )
    source = _normalize_month(source.filter(pl.col("keyword").is_in(sorted(EXPECTED_TEXT_SIGNALS))))
    return source.with_columns(
        [
            pl.col("keyword").alias("keyword_family"),
            pl.col("keyword_fa").alias("keyword_display_name"),
            pl.col("population_n").cast(pl.Int64, strict=False).alias("sale_valid_listing_count"),
            pl.col("positive_n").cast(pl.Int64, strict=False).alias("matched_listing_count"),
            pl.col("positive_rate").cast(pl.Float64, strict=False).alias("matched_rate"),
            (pl.col("positive_rate").cast(pl.Float64, strict=False) * 100.0).alias("matched_rate_pct"),
            pl.lit("sale").alias("price_regime"),
            pl.lit("regime:sale").alias("price_regime_key"),
            pl.lit(version).alias("method_version"),
        ]
    ).select(
        [
            "analysis_month", "month_key", "keyword_family", "keyword_display_name",
            "sale_valid_listing_count", "matched_listing_count", "matched_rate", "matched_rate_pct",
            "price_regime", "price_regime_key", "method_version",
        ]
    ).sort(["analysis_month", "keyword_family"])


def _segment_positioning() -> pl.DataFrame:
    """Reconstruct the frozen SEG-27 relative-price positioning without mutating M3.

    Current SEG-27 keeps ``listing_segments.parquet`` intentionally lightweight: it stores
    row lineage, the frozen market-type assignment, and the city/cat3 context, while the
    accepted asking-price metric remains in ``analysis_ready_features.parquet``.  Gold joins
    the two by ``source_row_id`` and reproduces the SEG-27 compatible price-reference
    hierarchy used by the rule-based typology: city x cat3 sale median when sufficiently
    populated, then national cat3 sale median, then the global sale median.  Neighborhood
    is deliberately not used in this normalization.
    """
    target = "sale_price_per_sqm_final_toman"
    reference_min_n = int(setting("milestone_3", "segmentation", "reference_min_n", default=30))
    if reference_min_n < 1:
        raise ValueError("milestone_3.segmentation.reference_min_n must be >= 1")

    assignments = pl.scan_parquet(SOURCES["segment_assignments"])
    assignment_required = {"source_row_id", "segment_id", "city_slug", "cat3_slug"}
    assignment_available = set(assignments.collect_schema().names())
    assignment_missing = sorted(assignment_required - assignment_available)
    if assignment_missing:
        raise ValueError(
            f"listing_segments.parquet is missing SEG-27 handoff columns: {assignment_missing}"
        )
    assignments = assignments.select(
        ["source_row_id", "segment_id", "city_slug", "cat3_slug"]
    )

    features = pl.scan_parquet(SOURCES["analysis_ready_features"])
    feature_required = {"source_row_id", target}
    feature_available = set(features.collect_schema().names())
    feature_missing = sorted(feature_required - feature_available)
    if feature_missing:
        raise ValueError(
            f"analysis_ready_features.parquet is missing SEG-27 positioning columns: {feature_missing}"
        )
    features = features.select(["source_row_id", target])

    assignment_stats = assignments.select(
        [
            pl.len().alias("row_n"),
            pl.col("source_row_id").n_unique().alias("unique_id_n"),
            pl.col("source_row_id").null_count().alias("null_id_n"),
        ]
    ).collect(engine="streaming").row(0, named=True)
    if assignment_stats["null_id_n"] or assignment_stats["row_n"] != assignment_stats["unique_id_n"]:
        raise ValueError("listing_segments.parquet must contain one non-null row per source_row_id")

    feature_stats = features.select(
        [
            pl.len().alias("row_n"),
            pl.col("source_row_id").n_unique().alias("unique_id_n"),
            pl.col("source_row_id").null_count().alias("null_id_n"),
        ]
    ).collect(engine="streaming").row(0, named=True)
    if feature_stats["null_id_n"] or feature_stats["row_n"] != feature_stats["unique_id_n"]:
        raise ValueError("analysis_ready_features.parquet must contain one non-null row per source_row_id")

    price = pl.col(target).cast(pl.Float64, strict=False)
    valid_price = price.is_not_null() & price.is_finite() & (price > 0)
    work = (
        assignments.join(features, on="source_row_id", how="left")
        .with_columns(
            [
                pl.col("city_slug").cast(pl.String, strict=False).fill_null("__missing_city__"),
                pl.col("cat3_slug").cast(pl.String, strict=False).fill_null("__missing_cat3__"),
                price.alias(target),
            ]
        )
    )
    handoff = work.select(
        [
            pl.len().alias("row_n"),
            valid_price.fill_null(False).sum().alias("valid_price_n"),
        ]
    ).collect(engine="streaming").row(0, named=True)
    if handoff["row_n"] != assignment_stats["row_n"] or handoff["valid_price_n"] != handoff["row_n"]:
        raise ValueError(
            "SEG-27 assignment-to-analysis-ready handoff is incomplete or contains invalid sale asking PSM values"
        )

    global_median = work.select(price.median().alias("global_median")).collect(engine="streaming").item()
    if global_median is None or not float(global_median) > 0:
        raise ValueError("SEG-27 positioning could not derive a positive global sale median")

    city_cat3_ref = work.group_by(["city_slug", "cat3_slug"]).agg(
        [
            pl.len().alias("city_cat3_n"),
            price.median().alias("city_cat3_median"),
        ]
    )
    cat3_ref = work.group_by("cat3_slug").agg(
        [
            pl.len().alias("cat3_n"),
            price.median().alias("cat3_median"),
        ]
    )

    scored = (
        work.join(city_cat3_ref, on=["city_slug", "cat3_slug"], how="left")
        .join(cat3_ref, on="cat3_slug", how="left")
        .with_columns(
            pl.when(pl.col("city_cat3_n") >= reference_min_n)
            .then(pl.col("city_cat3_median"))
            .when(pl.col("cat3_n") >= reference_min_n)
            .then(pl.col("cat3_median"))
            .otherwise(pl.lit(float(global_median)))
            .alias("segmentation_price_reference_psm")
        )
        .with_columns(
            (price / pl.col("segmentation_price_reference_psm") * 100.0)
            .alias("normalized_price_index")
        )
    )

    return (
        scored.filter(pl.col("segment_id").is_not_null())
        .group_by("segment_id")
        .agg(
            [
                pl.len().alias("positioning_listing_n"),
                pl.col("normalized_price_index")
                .median()
                .alias("median_normalized_price_index"),
            ]
        )
        .collect(engine="streaming")
    )


def build_segment_profile(manifest: dict[str, Any]) -> pl.DataFrame:
    source = _read_csv(SOURCES["segment_profile"])
    _require_columns(
        source,
        [
            "segment_id", "segment_name", "segment_method", "listing_n", "listing_share_pct",
            "dominant_property_family", "median_sale_price_per_sqm_final_toman",
            "median_primary_area_sqm", "median_rooms_count_num", "median_building_age_years",
            "amenity_index_pct", "dominant_property_type", "representative_cities", "applicability_note",
        ],
        "segment_profile.csv",
    )
    positioning = _segment_positioning()
    selected_ari = manifest.get("selected_stability_ari")
    primary_method = str(manifest.get("primary_method", "unknown"))
    return (
        source.join(positioning, on="segment_id", how="left")
        .with_columns(
            [
                pl.col("segment_method").alias("method_type"),
                pl.col("dominant_property_family").alias("property_family"),
                pl.concat_str([pl.lit("family:"), pl.col("dominant_property_family").cast(pl.String)]).alias("property_key"),
                pl.col("median_sale_price_per_sqm_final_toman").alias("median_asking_price_per_sqm_toman"),
                pl.col("median_primary_area_sqm").alias("median_area_sqm"),
                pl.col("median_rooms_count_num").alias("median_rooms"),
                pl.col("median_building_age_years").alias("median_age_years"),
                pl.col("amenity_index_pct").alias("amenity_profile_pct"),
                pl.col("applicability_note").alias("profile_applicability"),
                pl.lit(selected_ari, dtype=pl.Float64).alias("selected_stability_ari"),
                pl.lit(primary_method != "rule_based_descriptive_typology").alias("stability_ari_applicable_flag"),
                pl.lit(_source_version(manifest)).alias("method_version"),
            ]
        )
        .select(
            [
                "segment_id", "segment_name", "method_type", "property_family", "property_key",
                "listing_n", "listing_share_pct", "positioning_listing_n",
                "median_asking_price_per_sqm_toman", "median_area_sqm", "median_normalized_price_index",
                "median_rooms", "median_age_years", "amenity_profile_pct", "dominant_property_type",
                "representative_cities", "profile_applicability", "selected_stability_ari",
                "stability_ari_applicable_flag", "method_version",
            ]
        )
        .sort("segment_id")
    )

def build_segment_monthly_mix(manifest: dict[str, Any]) -> pl.DataFrame:
    assignments = pl.scan_parquet(SOURCES["segment_assignments"])
    available = set(assignments.collect_schema().names())
    required = {"analysis_month", "segment_id"}
    missing = sorted(required - available)
    if missing:
        raise ValueError(f"listing_segments.parquet is missing monthly-mix columns: {missing}")
    monthly = (
        assignments.select(["analysis_month", "segment_id"])
        .filter(pl.col("segment_id").is_not_null() & pl.col("analysis_month").is_not_null())
        .with_columns(pl.col("analysis_month").cast(pl.String).str.slice(0, 7).alias("analysis_month"))
        .group_by(["analysis_month", "segment_id"])
        .agg(pl.len().alias("listing_n"))
        .with_columns(pl.col("listing_n").sum().over("analysis_month").alias("month_total_n"))
        .with_columns((pl.col("listing_n") / pl.col("month_total_n") * 100.0).alias("listing_share_pct"))
        .collect(engine="streaming")
    )
    profile = _read_csv(SOURCES["segment_profile"]).select(
        ["segment_id", "segment_name", "segment_method"]
    ).unique("segment_id")
    return (
        _normalize_month(monthly)
        .join(profile, on="segment_id", how="left")
        .with_columns(
            [
                pl.col("segment_method").alias("method_type"),
                pl.lit(_source_version(manifest)).alias("method_version"),
            ]
        )
        .select(
            [
                "analysis_month", "month_key", "segment_id", "segment_name", "method_type",
                "listing_n", "month_total_n", "listing_share_pct", "method_version",
            ]
        )
        .sort(["analysis_month", "segment_id"])
    )

def build_location_dimension(*frames: pl.DataFrame) -> pl.DataFrame:
    parts: list[pl.DataFrame] = []
    for frame in frames:
        if not {"location_key", "city_slug", "neighborhood_slug"}.issubset(frame.columns):
            continue
        parts.append(
            frame.select(
                [
                    pl.col("location_key").cast(pl.String, strict=False),
                    pl.col("city_slug").cast(pl.String, strict=False),
                    pl.col("neighborhood_slug").cast(pl.String, strict=False),
                ]
            ).unique()
        )
    if not parts:
        raise ValueError("No location-bearing Gold marts were available.")

    combined = pl.concat(parts, how="diagonal_relaxed").filter(pl.col("location_key").is_not_null())
    if combined.height == 0:
        raise ValueError("Location dimension has no non-null location keys.")

    conflicts = (
        combined.group_by("location_key")
        .agg(
            [
                pl.col("city_slug").drop_nulls().n_unique().alias("city_count"),
                pl.col("neighborhood_slug").drop_nulls().n_unique().alias("neighborhood_count"),
            ]
        )
        .filter((pl.col("city_count") > 1) | (pl.col("neighborhood_count") > 1))
    )
    if conflicts.height:
        raise ValueError(
            "Location key maps to conflicting city/neighborhood values: "
            f"{conflicts.head(10).to_dicts()}"
        )

    out = (
        combined.group_by("location_key", maintain_order=True)
        .agg(
            [
                pl.col("city_slug").drop_nulls().first().alias("city_slug"),
                pl.col("neighborhood_slug").drop_nulls().first().alias("neighborhood_slug"),
            ]
        )
        .sort("location_key")
    )
    rows = out.to_dicts()
    for row in rows:
        key = str(row.get("location_key") or "")
        row["location_level"] = (
            "national" if key == "NATIONAL" else ("city" if key.startswith("CITY:") else "neighborhood")
        )
        row["city_key"] = (
            None
            if row["location_level"] == "national" or not row.get("city_slug")
            else f"CITY:{row['city_slug']}"
        )
        row["city_display_name"] = _humanize(row.get("city_slug"))
        row["neighborhood_display_name"] = _humanize(row.get("neighborhood_slug"))
        row["is_four_city"] = row.get("city_slug") in FOUR_CITY_SLUGS
        if row["location_level"] == "national":
            row["location_display_name"] = "National"
            row["location_sort_key"] = "0|NATIONAL"
        elif row["location_level"] == "city":
            row["location_display_name"] = row["city_display_name"]
            row["location_sort_key"] = f"1|{row.get('city_slug') or ''}"
        else:
            row["location_display_name"] = f"{row['city_display_name']} - {row['neighborhood_display_name']}"
            row["location_sort_key"] = f"2|{row.get('city_slug') or ''}|{row.get('neighborhood_slug') or ''}"
    location_schema = {
        "location_key": pl.String,
        "city_slug": pl.String,
        "neighborhood_slug": pl.String,
        "location_level": pl.String,
        "city_key": pl.String,
        "city_display_name": pl.String,
        "neighborhood_display_name": pl.String,
        "is_four_city": pl.Boolean,
        "location_display_name": pl.String,
        "location_sort_key": pl.String,
    }
    return pl.DataFrame(rows, schema=location_schema, strict=False).sort("location_sort_key")


def build_month_dimension(*frames: pl.DataFrame) -> pl.DataFrame:
    months: list[pl.DataFrame] = []
    for frame in frames:
        if {"analysis_month", "month_key"}.issubset(frame.columns):
            months.append(
                frame.select(
                    [
                        pl.col("analysis_month").cast(pl.String).str.slice(0, 7),
                        pl.col("month_key").cast(pl.Int32, strict=False),
                    ]
                ).drop_nulls().unique()
            )
    if not months:
        raise ValueError("No month-bearing Gold marts were available.")
    out = pl.concat(months, how="diagonal_relaxed").unique()
    conflicts = (
        out.group_by("month_key")
        .agg(pl.col("analysis_month").n_unique().alias("month_label_count"))
        .filter(pl.col("month_label_count") > 1)
    )
    if conflicts.height:
        raise ValueError(
            "Month dimension contains conflicting labels for the same month_key: "
            f"{conflicts.head(10).to_dicts()}"
        )
    return (
        out.sort("month_key")
        .with_columns(
            [
                pl.concat_str([pl.col("analysis_month"), pl.lit("-01")])
                .str.strptime(pl.Date, format="%Y-%m-%d", strict=True)
                .alias("month_start"),
                (pl.col("month_key") // 100).cast(pl.Int32).alias("year"),
                (pl.col("month_key") % 100).cast(pl.Int8).alias("month_number"),
                (((pl.col("month_key") % 100) - 1) // 3 + 1).cast(pl.Int8).alias("quarter_number"),
                pl.col("analysis_month").alias("month_label"),
                pl.col("month_key").rank(method="dense").cast(pl.Int32).alias("chronological_sort"),
                pl.lit(True).alias("core_period_flag"),
            ]
        )
        .sort("month_key")
    )


def build_property_dimension(*frames: pl.DataFrame) -> pl.DataFrame:
    mappings = load_domain_mappings().get("property_families", {})
    observed_keys: set[str] = set()
    observed_family_by_key: dict[str, str | None] = {}
    for frame in frames:
        if "property_key" not in frame.columns:
            continue
        columns = ["property_key"] + (["property_family"] if "property_family" in frame.columns else [])
        for row in frame.select(columns).drop_nulls(subset=["property_key"]).unique().to_dicts():
            key = str(row["property_key"])
            observed_keys.add(key)
            observed_family_by_key[key] = row.get("property_family")

    rows: list[dict[str, Any]] = []
    for family, spec in mappings.items():
        if not isinstance(spec, dict):
            continue
        family_key = f"family:{family}"
        rows.append(
            {
                "property_key": family_key,
                "property_family": str(family),
                "property_category": None,
                "property_type": None,
                "property_display_name": str(spec.get("label_fa") or _humanize(family) or family),
                "property_level": "family",
                "is_observed_in_gold": family_key in observed_keys,
            }
        )
        for category in spec.get("categories", []) or []:
            category_key = f"category:{category}"
            rows.append(
                {
                    "property_key": category_key,
                    "property_family": str(family),
                    "property_category": str(category),
                    "property_type": None,
                    "property_display_name": _humanize(category),
                    "property_level": "category",
                    "is_observed_in_gold": category_key in observed_keys,
                }
            )

    canonical_keys = {str(row["property_key"]) for row in rows}
    for key in sorted(observed_keys - canonical_keys):
        prefix, _, token = key.partition(":")
        family = observed_family_by_key.get(key) or (token if prefix == "family" else None)
        rows.append(
            {
                "property_key": key,
                "property_family": family,
                "property_category": token if prefix == "category" else None,
                "property_type": None,
                "property_display_name": _humanize(token) or token,
                "property_level": prefix or "unknown",
                "is_observed_in_gold": True,
            }
        )
    if not rows:
        raise ValueError("Property dimension has no canonical or observed values.")
    return (
        pl.DataFrame(rows)
        .unique(subset=["property_key"], keep="first")
        .with_columns(
            pl.when(pl.col("property_level") == "family")
            .then(pl.lit(1))
            .when(pl.col("property_level") == "category")
            .then(pl.lit(2))
            .otherwise(pl.lit(9))
            .alias("property_level_sort")
        )
        .sort(["property_family", "property_level_sort", "property_category"], nulls_last=True)
    )


def build_price_regime_dimension(*frames: pl.DataFrame) -> pl.DataFrame:
    regimes = load_domain_mappings().get("price_regimes", {})
    canonical_values: set[str] = set()
    if isinstance(regimes, dict):
        for key, value in regimes.items():
            if isinstance(value, str):
                canonical_values.add(value)
            elif isinstance(key, str):
                canonical_values.add(key)

    observed_values: set[str] = set()
    for frame in frames:
        if "price_regime" in frame.columns:
            observed_values.update(frame.get_column("price_regime").drop_nulls().cast(pl.String).to_list())
        elif "price_regime_key" in frame.columns:
            observed_values.update(
                value.split(":", 1)[-1]
                for value in frame.get_column("price_regime_key").drop_nulls().cast(pl.String).to_list()
            )
    values = canonical_values | observed_values
    if not values:
        raise ValueError("Price-regime dimension has no canonical or observed values.")
    return pl.DataFrame(
        [
            {
                "price_regime_key": f"regime:{regime}",
                "price_regime": regime,
                "price_regime_display_name": _humanize(regime),
                "is_observed_in_gold": regime in observed_values,
                "price_observation_type": price_observation_type(),
                "price_unit": price_unit(),
            }
            for regime in sorted(values)
        ]
    ).sort("price_regime_key")


def build_user_type_dimension(seller: pl.DataFrame) -> pl.DataFrame:
    _require_columns(seller, ["agency_label", "personal_label"], "mart_seller_type")
    values = []
    for column in ["agency_label", "personal_label"]:
        values.extend(seller.get_column(column).drop_nulls().cast(pl.String).unique().to_list())
    rows = [
        {
            "user_type_key": f"user:{value}",
            "user_type": value,
            "user_type_display_name": value,
        }
        for value in sorted(set(values))
    ]
    return pl.DataFrame(rows)


def _first_existing_expr(
    frame: pl.DataFrame,
    candidates: list[str],
    *,
    dtype: pl.DataType = pl.String,
) -> pl.Expr:
    available = [column for column in candidates if column in frame.columns]
    if not available:
        return pl.lit(None, dtype=dtype)
    return pl.coalesce([pl.col(column).cast(dtype, strict=False) for column in available])


def _model_quality_key_expr() -> pl.Expr:
    fields = [
        "record_type", "model_name", "population", "evaluation_split", "evaluation_stage",
        "error_scope", "city_slug", "property_family", "benchmark_role",
    ]
    return pl.concat_str(
        [pl.col(field).cast(pl.String, strict=False).fill_null("__ALL__") for field in fields],
        separator="|",
    )


def _mart_source_metadata(name: str) -> tuple[str, str, str, str]:
    mapping = {
        "mart_market_monthly": (
            "monthly_market", "monthly_market_summary.csv + monthly_supply_summary.csv",
            "Monthly platform listing activity and apartment-sale asking-price series.",
            "Listing activity is a platform-flow proxy, not physical inventory, liquidity, or absorption.",
        ),
        "mart_location_market": (
            "market_map + market_temperature", "neighborhood_market_summary.csv + market_temperature_summary.csv",
            "All eligible canonical locations with neighborhood asking-price context and/or all-city listing-market temperature.",
            "Asking prices are not transactions; Market Temperature is a relative listing-market proxy, not liquidity or absorption.",
        ),
        "mart_price_driver_effects": (
            "price_drivers", "price_driver_summary.csv", "Frozen all-sales adjusted association model population.",
            "Model-implied observational association; not a causal effect or price share.",
        ),
        "mart_price_driver_importance": (
            "price_drivers", "price_driver_permutation_importance.csv", "Frozen held-out predictive contribution population.",
            "Permutation importance is predictive contribution, not causal importance.",
        ),
        "mart_model_quality": (
            "price_drivers", "price_driver_model_diagnostics.csv + price_model_benchmark.csv + avm_error_analysis.csv",
            "Frozen held-out model diagnostics and AVM error-analysis populations.",
            "Research prototype diagnostics; not a production valuation guarantee.",
        ),
        "mart_seller_type": (
            "seller_type_comparison", "seller_type_comparison_summary.csv", "Frozen agency-versus-personal comparable-sale population.",
            "Adjusted seller difference is observational and does not establish a causal agency premium.",
        ),
        "mart_text_signals": (
            "text_analysis", "text_signal_summary.csv", "Validated sale-listing text-signal population.",
            "Fixed validated association; BI slicers must not imply model refitting.",
        ),
        "mart_text_monthly": (
            "text_analysis", "text_keyword_monthly_frequency.csv", "Monthly frequency of the six validated text-signal families.",
            "Frequency is descriptive and does not change the fixed adjusted text estimates.",
        ),
        "mart_segment_profile": (
            "market_segmentation", "segment_profile.csv + listing_segments.parquet + price_drivers/analysis_ready_features.parquet", "Frozen sale-market types / descriptive typology profile with SEG-27-compatible relative-price positioning reconstructed by source_row_id handoff.",
            "Fallback market types are descriptive and must not be relabeled as stable statistical clusters.",
        ),
        "mart_segment_monthly_mix": (
            "market_segmentation", "listing_segments.parquet", "Monthly mix aggregated from frozen listing-level segment assignments.",
            "Monthly shares describe the observed listing population and do not imply transitions or causality.",
        ),
    }
    return mapping[name]

def _sample_n_expr(name: str, frame: pl.DataFrame) -> pl.Expr:
    candidates = {
        "mart_market_monthly": ["price_listing_n", "deduplicated_listing_count"],
        "mart_location_market": ["valid_price_n", "temperature_sample_n", "listing_n"],
        "mart_price_driver_effects": ["sample_n"],
        "mart_price_driver_importance": ["sample_n"],
        "mart_model_quality": ["n"],
        "mart_seller_type": ["sample_size", "crossfit_rows"],
        "mart_text_signals": ["sample_n", "heldout_test_n"],
        "mart_text_monthly": ["sale_valid_listing_count"],
        "mart_segment_profile": ["listing_n"],
        "mart_segment_monthly_mix": ["listing_n"],
    }[name]
    return _first_existing_expr(frame, candidates, dtype=pl.Int64)

def _reliability_status_expr(name: str, frame: pl.DataFrame) -> pl.Expr:
    if name == "mart_market_monthly" and "price_reliable_flag" in frame.columns:
        value = (
            pl.when(pl.col("price_reliable_flag").is_null()).then(pl.lit("descriptive"))
            .when(pl.col("price_reliable_flag") == True).then(pl.lit("reliable"))
            .otherwise(pl.lit("review"))
        )
    elif name == "mart_location_market":
        value = (
            pl.when(
                (pl.col("price_map_available_flag") == True) & (pl.col("reliable_flag") == True)
            ).then(pl.lit("reliable"))
            .when(
                (pl.col("temperature_available_flag") == True)
                & (pl.col("temperature_reliability_eligible_flag") == True)
            ).then(pl.lit("reliable"))
            .otherwise(pl.lit("review"))
        )
    elif "reliability_status" in frame.columns:
        default = "descriptive" if name in {"mart_segment_profile", "mart_segment_monthly_mix", "mart_text_monthly"} else "validated"
        value = pl.coalesce([_clean_text_expr("reliability_status"), pl.lit(default)])
    elif name in {"mart_segment_profile", "mart_segment_monthly_mix", "mart_text_monthly"}:
        value = pl.lit("descriptive")
    else:
        value = pl.lit("validated")
    return value.cast(pl.String, strict=False).str.strip_chars().str.to_lowercase()

def _reliability_reason_expr(name: str, frame: pl.DataFrame) -> pl.Expr:
    if name == "mart_market_monthly" and "price_reliable_flag" in frame.columns:
        default = (
            pl.when(pl.col("price_reliable_flag").is_null()).then(pl.lit("Supply-only descriptive row."))
            .when(pl.col("price_reliable_flag") == True).then(pl.lit("Upstream price reliability gate passed."))
            .otherwise(pl.lit("Upstream price reliability gate requires review."))
        )
    elif name == "mart_location_market":
        default = (
            pl.when((pl.col("price_map_available_flag") == True) & (pl.col("reliable_flag") == True))
            .then(pl.lit("Upstream neighborhood price reliability gate passed."))
            .when((pl.col("temperature_available_flag") == True) & (pl.col("temperature_reliability_eligible_flag") == True))
            .then(pl.lit("All-city temperature month/sample reliability gate passed."))
            .otherwise(pl.lit("Location retained for transparent exploration with reliability limitations."))
        )
    else:
        default = pl.lit("Accepted frozen upstream output; no Gold-layer retesting or refit.")
    if "reliability_reason" in frame.columns:
        return pl.coalesce([_clean_text_expr("reliability_reason"), default])
    return default

def standardize_mart(name: str, frame: pl.DataFrame) -> pl.DataFrame:
    """Apply shared Gold semantics without changing accepted analytical estimates."""
    if name not in EXPECTED_MARTS:
        raise KeyError(f"Unknown Gold mart: {name}")
    source_task_id, source_artifact, population_definition, limitation_note = _mart_source_metadata(name)
    out = frame
    if name == "mart_model_quality":
        required = [
            "record_type", "model_name", "population", "evaluation_split", "evaluation_stage",
            "error_scope", "city_slug", "property_family", "benchmark_role",
        ]
        _require_columns(out, required, name)
        out = out.with_columns(_model_quality_key_expr().alias("model_quality_key"))

    key_columns = {
        "location_key", "property_key", "price_regime_key", "user_type_key",
        "feature_id", "feature_or_block_id", "model_quality_key", "comparison", "keyword_family",
    }
    casts: list[pl.Expr] = [
        pl.col(column).cast(pl.String, strict=False).alias(column)
        for column in key_columns.intersection(out.columns)
    ]
    if "month_key" in out.columns:
        casts.append(pl.col("month_key").cast(pl.Int32, strict=False).alias("month_key"))
    if casts:
        out = out.with_columns(casts)

    method_version_expr = _first_existing_expr(
        out, ["method_version", "analysis_version"], dtype=pl.String
    ).fill_null("unknown")
    if name in {"mart_segment_monthly_mix", "mart_text_monthly"}:
        price_observation_expr = pl.lit(None, dtype=pl.String)
        price_unit_expr = pl.lit(None, dtype=pl.String)
    elif name == "mart_market_monthly" and "series_kind" in out.columns:
        price_observation_expr = (
            pl.when(pl.col("series_kind") == "price_and_supply")
            .then(pl.lit(price_observation_type(), dtype=pl.String))
            .otherwise(pl.lit(None, dtype=pl.String))
        )
        price_unit_expr = (
            pl.when(pl.col("series_kind") == "price_and_supply")
            .then(pl.lit(price_unit(), dtype=pl.String))
            .otherwise(pl.lit(None, dtype=pl.String))
        )
    else:
        price_observation_expr = pl.lit(price_observation_type(), dtype=pl.String)
        price_unit_expr = pl.lit(price_unit(), dtype=pl.String)

    out = out.with_columns(
        [
            pl.lit(source_task_id).alias("source_task_id"),
            pl.lit(source_artifact).alias("source_artifact"),
            pl.lit(population_definition).alias("population_definition"),
            _reliability_status_expr(name, out).alias("reliability_status"),
            _reliability_reason_expr(name, out).alias("reliability_reason"),
            pl.lit(limitation_note).alias("limitation_note"),
            method_version_expr.alias("method_version"),
            _sample_n_expr(name, out).alias("sample_n"),
            price_observation_expr.alias("price_observation_type"),
            price_unit_expr.alias("price_unit"),
        ]
    )
    base_columns = [column for column in out.columns if column not in COMMON_MART_METADATA_COLUMNS]
    return out.select([*base_columns, *COMMON_MART_METADATA_COLUMNS])


def standardize_dimension(name: str, frame: pl.DataFrame) -> pl.DataFrame:
    key = {
        "dim_location": "location_key", "dim_month": "month_key", "dim_property": "property_key",
        "dim_price_regime": "price_regime_key", "dim_user_type": "user_type_key",
    }[name]
    if key == "month_key":
        return frame.with_columns(pl.col(key).cast(pl.Int32, strict=False).alias(key))
    return frame.with_columns(pl.col(key).cast(pl.String, strict=False).alias(key))

def build_artifact_contract() -> pl.DataFrame:
    return pl.DataFrame(artifact_contract_rows()).sort(["artifact_type", "artifact_name"])


def build_dashboard_page_registry() -> pl.DataFrame:
    return pl.DataFrame(dashboard_page_rows()).sort("page_order")


def build_gold_relationship_contract() -> pl.DataFrame:
    return pl.DataFrame(relationship_contract_rows()).sort("relationship_id")


def _reset_dashboard_metadata() -> list[str]:
    removed: list[str] = []
    for filename in DASHBOARD_METADATA_FILENAMES:
        path = METADATA_DIR / filename
        if path.exists():
            path.unlink()
            removed.append(relative_to_project(path))
    return removed


def _reset_gold_structure(*, keep_csv_copies: bool = False) -> list[str]:
    """Remove stale generated tables and enforce the canonical 10+5 file set."""
    removed: list[str] = []
    expected_by_directory = {
        MARTS_DIR: {
            *{f"{name}.parquet" for name in EXPECTED_MARTS},
            *({f"{name}.csv" for name in EXPECTED_MARTS} if keep_csv_copies else set()),
        },
        DIMENSIONS_DIR: {
            *{f"{name}.parquet" for name in EXPECTED_DIMENSIONS},
            *({f"{name}.csv" for name in EXPECTED_DIMENSIONS} if keep_csv_copies else set()),
        },
    }
    for directory, expected in expected_by_directory.items():
        if not directory.exists():
            continue
        for path in directory.iterdir():
            if not path.is_file() or path.suffix.lower() not in {".parquet", ".csv"}:
                continue
            is_legacy = path.stem in LEGACY_ARTIFACT_NAMES
            is_unexpected = path.name not in expected
            if is_legacy or is_unexpected:
                path.unlink()
                removed.append(relative_to_project(path))
    return sorted(removed)


def _inventory_rows(kind: str, frames: dict[str, pl.DataFrame], base_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, frame in frames.items():
        rows.append(
            {
                "artifact_type": kind,
                "artifact_name": name,
                "path": relative_to_project(base_dir / f"{name}.parquet"),
                "row_count": frame.height,
                "column_count": frame.width,
                "columns": ";".join(frame.columns),
            }
        )
    return rows


def _validate_no_coordinate_columns(frames: dict[str, pl.DataFrame]) -> list[str]:
    violations: list[str] = []
    for name, frame in frames.items():
        for column in frame.columns:
            lowered = column.lower()
            if any(token in lowered for token in FORBIDDEN_COLUMN_TOKENS):
                violations.append(f"{name}.{column}")
    return sorted(violations)


def _manifest_status_row(stage: str, component: str, path: Path, payload: dict[str, Any] | None) -> dict[str, Any]:
    if payload is None:
        return {
            "stage": stage, "component": component, "overall_status": "MISSING", "ready": False,
            "critical_failures": None, "review_count": None, "reliability_status": "missing",
            "sample_n": None, "source_manifest": relative_to_project(path),
        }
    status = payload.get("status") if isinstance(payload.get("status"), dict) else {}
    overall = str(status.get("overall_status", payload.get("overall_status", "UNKNOWN"))).upper()
    critical = status.get("critical_failures", payload.get("critical_failures", payload.get("critical_failure_count", 0)))
    review = status.get("review_count", payload.get("review_count", payload.get("warnings", payload.get("warning_count", 0))))
    ready_candidates = [
        status.get("ready"), payload.get("ready"), payload.get("m2_entry_ready"), payload.get("m3_entry_ready"),
        payload.get("ready_for_phase_2"), payload.get("freeze_authorized"), payload.get("gold_data_ready"),
    ]
    ready_values = [value for value in ready_candidates if isinstance(value, bool)]
    ready = any(ready_values) if ready_values else overall in {"PASS", "PASSED", "PASSED_WITH_DOCUMENTED_ASSUMPTIONS"}
    if _safe_int(critical, 0) > 0:
        ready = False
    sample_n = None
    for key in ("row_count", "population_rows", "analysis_population_size", "fit_assigned_rows"):
        if key in payload:
            sample_n = payload.get(key)
            break
    return {
        "stage": stage, "component": component, "overall_status": overall, "ready": bool(ready),
        "critical_failures": critical, "review_count": review,
        "reliability_status": "ready" if ready else ("review" if _safe_int(critical, 0) == 0 else "fail"),
        "sample_n": sample_n, "source_manifest": relative_to_project(path),
    }


def _tabular_quality_status_row(
    stage: str, component: str, path: Path
) -> dict[str, Any]:
    """Adapt a canonical PASS/REVIEW/FAIL quality table to dashboard status metadata.

    This is used for the current M2 closeout, where the accepted status artifact is a
    compact CSV rather than a JSON manifest. Readiness and review state remain separate:
    non-critical REVIEW rows do not become failures.
    """
    if not path.exists():
        return _manifest_status_row(stage, component, path, None)

    frame = _read_csv(path)
    _require_columns(frame, ["status", "critical"], path.name)
    records = frame.to_dicts()
    if not records:
        return {
            "stage": stage,
            "component": component,
            "overall_status": "FAIL",
            "ready": False,
            "critical_failures": 1,
            "review_count": 0,
            "reliability_status": "fail",
            "sample_n": None,
            "source_manifest": relative_to_project(path),
        }

    def as_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        return str(value or "").strip().lower() in {"true", "1", "yes", "y"}

    pass_statuses = {"PASS", "PASSED"}
    critical_failures = 0
    review_count = 0
    for row in records:
        status = str(row.get("status") or "").strip().upper()
        critical = as_bool(row.get("critical"))
        if critical and status not in pass_statuses:
            critical_failures += 1
        elif status not in pass_statuses:
            review_count += 1

    ready = critical_failures == 0
    overall = "FAIL" if critical_failures else ("REVIEW" if review_count else "PASS")

    sample_n: int | None = None
    if "check" in frame.columns and "actual" in frame.columns:
        for row in records:
            if str(row.get("check") or "") in {
                "source_row_id_unique",
                "dataset_rows",
                "row_count",
            }:
                try:
                    sample_n = int(float(str(row.get("actual")).replace(",", "")))
                except (TypeError, ValueError):
                    sample_n = None
                if sample_n is not None:
                    break

    return {
        "stage": stage,
        "component": component,
        "overall_status": overall,
        "ready": ready,
        "critical_failures": critical_failures,
        "review_count": review_count,
        "reliability_status": (
            "fail" if not ready else ("review" if review_count else "ready")
        ),
        "sample_n": sample_n,
        "source_manifest": relative_to_project(path),
    }


def build_dashboard_quality_status(
    manifests: dict[str, dict[str, Any]], build_status: dict[str, Any]
) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for stage, component, key in [
        ("M3", "Monthly Market", "monthly"), ("M3", "Market Map", "market_map"),
        ("M3", "Market Temperature", "temperature"), ("M3", "Price Drivers / AVM", "price_drivers"),
        ("M3", "Seller Type", "seller"), ("M3", "Text Signals", "text"),
        ("M3", "Market Segmentation", "segments"),
    ]:
        path_key = {
            "monthly": "monthly_manifest", "market_map": "market_map_manifest",
            "temperature": "market_temperature_manifest", "price_drivers": "price_driver_manifest",
            "seller": "seller_manifest", "text": "text_manifest", "segments": "segment_manifest",
        }[key]
        rows.append(_manifest_status_row(stage, component, SOURCES[path_key], manifests[key]))
    for stage, component, key in [
        ("M1", "Data Contract and Audit", "m1_closeout"),
        ("M2", "Price Quality Gate", "m2_quality_gate"),
    ]:
        path = QUALITY_STATUS_SOURCES[key]
        if path.suffix.lower() == ".csv":
            rows.append(_tabular_quality_status_row(stage, component, path))
        else:
            payload = _load_json(path) if path.exists() else None
            rows.append(_manifest_status_row(stage, component, path, payload))
    build_payload = {"status": build_status, "gold_data_ready": bool(build_status.get("ready") and str(build_status.get("overall_status", "")).upper() == "PASS")}
    rows.append(_manifest_status_row("M4", "Gold Build", QA_DIR / "gold_manifest.json", build_payload))
    order = {"M1": 1, "M2": 2, "M3": 3, "M4": 4}
    return pl.DataFrame(rows).with_columns(pl.col("stage").replace_strict(order, default=99).alias("stage_order")).sort(["stage_order", "component"]).drop("stage_order")


def run(
    *, write_csv_copies: bool = False, reset_dashboard_metadata: bool = False,
    reset_gold_structure: bool = False,
) -> dict[str, Path]:
    started = time.perf_counter()
    for directory in [MARTS_DIR, DIMENSIONS_DIR, METADATA_DIR, QA_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    removed_gold_structure = _reset_gold_structure(keep_csv_copies=write_csv_copies) if reset_gold_structure else []
    removed_dashboard_metadata = _reset_dashboard_metadata() if reset_dashboard_metadata else []

    show_progress(0, "validating frozen M3 dependencies")
    manifests, checks = validate_upstream()

    show_progress(12, "building monthly and location marts")
    raw_marts: dict[str, pl.DataFrame] = {
        "mart_market_monthly": build_market_monthly(_source_version(manifests["monthly"])),
        "mart_location_market": build_location_market(manifests["market_map"], manifests["temperature"]),
    }
    show_progress(30, "building driver and model marts")
    price_version = _source_version(manifests["price_drivers"])
    raw_marts.update(
        {
            "mart_price_driver_effects": build_price_driver_effects(price_version),
            "mart_price_driver_importance": build_price_driver_importance(price_version),
            "mart_model_quality": build_model_quality(price_version),
        }
    )
    show_progress(50, "building seller, text, and segment marts")
    text_version = _source_version(manifests["text"])
    raw_marts.update(
        {
            "mart_seller_type": build_seller_type(_source_version(manifests["seller"])),
            "mart_text_signals": build_text_signals(text_version),
            "mart_text_monthly": build_text_monthly(text_version),
            "mart_segment_profile": build_segment_profile(manifests["segments"]),
            "mart_segment_monthly_mix": build_segment_monthly_mix(manifests["segments"]),
        }
    )
    marts = {name: standardize_mart(name, raw_marts[name]) for name in EXPECTED_MARTS}

    monthly_market_invalid = _invalid_location_grain_count(
        _read_csv(SOURCES["monthly_market"]), "monthly_market_summary.csv"
    )
    monthly_supply_invalid = _invalid_location_grain_count(
        _read_csv(SOURCES["monthly_supply"]), "monthly_supply_summary.csv"
    )
    gold_unmapped_locations = int(
        marts["mart_market_monthly"].get_column("location_key").null_count()
    )
    checks.append(
        make_check(
            "monthly_unmapped_location_buckets_excluded",
            "source_quality",
            (
                f"market_source_invalid={monthly_market_invalid};"
                f"supply_source_invalid={monthly_supply_invalid};"
                f"gold_null_location_keys={gold_unmapped_locations}"
            ),
            "gold_null_location_keys=0; source buckets are excluded, never relabeled",
            gold_unmapped_locations == 0,
            notes=(
                "Missing-city aggregate buckets cannot join dim_location. Gold preserves "
                "the separately published national aggregates and excludes only unmapped "
                "location-grain rows; it does not invent a city or relabel them as national."
            ),
        )
    )

    show_progress(70, "building five conformed dimensions")
    raw_dimensions: dict[str, pl.DataFrame] = {
        "dim_location": build_location_dimension(marts["mart_market_monthly"], marts["mart_location_market"]),
        "dim_month": build_month_dimension(
            marts["mart_market_monthly"], marts["mart_text_monthly"], marts["mart_segment_monthly_mix"]
        ),
        "dim_property": build_property_dimension(*marts.values()),
        "dim_price_regime": build_price_regime_dimension(*marts.values()),
        "dim_user_type": build_user_type_dimension(marts["mart_seller_type"]),
    }
    dimensions = {name: standardize_dimension(name, raw_dimensions[name]) for name in EXPECTED_DIMENSIONS}

    show_progress(82, "running structural Gold checks")
    for name in EXPECTED_MARTS:
        frame = marts[name]
        checks.append(make_check(f"{name}_nonempty", "gold_mart", frame.height, ">0", frame.height > 0))
        missing = sorted(set(REQUIRED_ARTIFACT_COLUMNS[name] + COMMON_MART_METADATA_COLUMNS) - set(frame.columns))
        checks.append(make_check(f"{name}_required_columns", "schema", missing, [], not missing))
        grain = list(MART_GRAIN_KEYS[name])
        duplicate_rows = int(frame.select(grain).is_duplicated().sum())
        checks.append(make_check(f"{name}_grain_unique", "grain", duplicate_rows, 0, duplicate_rows == 0))

    for name in EXPECTED_DIMENSIONS:
        frame = dimensions[name]
        checks.append(make_check(f"{name}_nonempty", "dimension", frame.height, ">0", frame.height > 0))
        missing = sorted(set(REQUIRED_ARTIFACT_COLUMNS[name]) - set(frame.columns))
        checks.append(make_check(f"{name}_required_columns", "schema", missing, [], not missing))

    forbidden = _validate_no_coordinate_columns({**marts, **dimensions})
    checks.append(make_check(
        "no_coordinates_or_legacy_semantics_in_gold", "privacy",
        ";".join(forbidden) if forbidden else "none", "none", not forbidden,
        notes="Gold carries no exact coordinates and no superseded tightness/legacy fields.",
    ))
    observed_keywords = set(marts["mart_text_signals"].get_column("keyword_family").cast(pl.String).to_list())
    monthly_keywords = set(marts["mart_text_monthly"].get_column("keyword_family").cast(pl.String).to_list())
    checks.append(make_check("validated_text_signal_family", "text", sorted(observed_keywords), sorted(EXPECTED_TEXT_SIGNALS), observed_keywords == set(EXPECTED_TEXT_SIGNALS)))
    checks.append(make_check("monthly_text_signal_family", "text", sorted(monthly_keywords), sorted(EXPECTED_TEXT_SIGNALS), monthly_keywords == set(EXPECTED_TEXT_SIGNALS)))
    segment_count_mismatches = marts["mart_segment_profile"].filter(
        pl.col("listing_n").cast(pl.Int64, strict=False) != pl.col("positioning_listing_n").cast(pl.Int64, strict=False)
    ).height
    checks.append(make_check("segment_profile_assignment_counts_match", "segments", segment_count_mismatches, 0, segment_count_mismatches == 0))

    show_progress(91, "writing canonical Gold artifacts")
    outputs: dict[str, Path] = {}
    for name, frame in marts.items():
        path = MARTS_DIR / f"{name}.parquet"
        _write_parquet(frame, path)
        if write_csv_copies:
            atomic_write_csv(frame, MARTS_DIR / f"{name}.csv")
        outputs[name] = path
    for name, frame in dimensions.items():
        path = DIMENSIONS_DIR / f"{name}.parquet"
        _write_parquet(frame, path)
        if write_csv_copies:
            atomic_write_csv(frame, DIMENSIONS_DIR / f"{name}.csv")
        outputs[name] = path

    artifact_contract = build_artifact_contract()
    relationship_contract = build_gold_relationship_contract()
    page_registry = build_dashboard_page_registry()
    artifact_contract_path = METADATA_DIR / "gold_artifact_contract.csv"
    relationship_contract_path = METADATA_DIR / "gold_relationship_contract.csv"
    page_registry_path = METADATA_DIR / "dashboard_page_registry.csv"
    quality_status_path = METADATA_DIR / "dashboard_quality_status.csv"
    lineage_path = METADATA_DIR / "gold_source_lineage.csv"
    inventory_path = METADATA_DIR / "gold_artifact_inventory.csv"
    checks_path = QA_DIR / "gold_checks.csv"
    manifest_path = QA_DIR / "gold_manifest.json"

    atomic_write_csv(artifact_contract, artifact_contract_path)
    atomic_write_csv(relationship_contract, relationship_contract_path)
    atomic_write_csv(page_registry, page_registry_path)
    lineage_rows = [
        {"source_name": name, "source_path": relative_to_project(path), "exists": path.exists(), "role": "frozen_m3_source"}
        for name, path in SOURCES.items()
    ] + [
        {"source_name": name, "source_path": relative_to_project(path), "exists": path.exists(), "role": "quality_status_source"}
        for name, path in QUALITY_STATUS_SOURCES.items()
    ]
    atomic_write_csv(pl.DataFrame(lineage_rows), lineage_path)

    atomic_write_csv(checks_frame(checks), checks_path)
    status = summarize_checks(checks)
    gold_data_ready = bool(status.get("ready") and str(status.get("overall_status", "")).upper() == "PASS")
    quality_status = build_dashboard_quality_status(manifests, status)
    atomic_write_csv(quality_status, quality_status_path)

    inventory = pl.DataFrame(
        _inventory_rows("mart", marts, MARTS_DIR)
        + _inventory_rows("dimension", dimensions, DIMENSIONS_DIR)
        + [
            {"artifact_type": "metadata", "artifact_name": "gold_artifact_contract", "path": relative_to_project(artifact_contract_path), "row_count": artifact_contract.height, "column_count": artifact_contract.width, "columns": ";".join(artifact_contract.columns)},
            {"artifact_type": "metadata", "artifact_name": "gold_relationship_contract", "path": relative_to_project(relationship_contract_path), "row_count": relationship_contract.height, "column_count": relationship_contract.width, "columns": ";".join(relationship_contract.columns)},
            {"artifact_type": "metadata", "artifact_name": "dashboard_page_registry", "path": relative_to_project(page_registry_path), "row_count": page_registry.height, "column_count": page_registry.width, "columns": ";".join(page_registry.columns)},
            {"artifact_type": "metadata", "artifact_name": "dashboard_quality_status", "path": relative_to_project(quality_status_path), "row_count": quality_status.height, "column_count": quality_status.width, "columns": ";".join(quality_status.columns)},
        ]
    )
    atomic_write_csv(inventory, inventory_path)

    atomic_write_json(
        {
            "version": VERSION, "status": status, "gold_data_ready": gold_data_ready,
            "ready_for_metadata_design": gold_data_ready,
            "dashboard_metadata_status": "PENDING_M4_05_GENERATION",
            "gold_structure_removed_on_request": removed_gold_structure,
            "dashboard_metadata_removed_on_request": removed_dashboard_metadata,
            "architecture": {
                "dashboard_pages": dashboard_page_rows(), "gold_marts": len(EXPECTED_MARTS),
                "conformed_dimensions": len(EXPECTED_DIMENSIONS), "physical_relationships": len(RELATIONSHIP_SPECS),
                "semantic_only_dimensions": sorted(SEMANTIC_ONLY_DIMENSIONS),
                "minimum_n_parameter": "disconnected BI parameter; not a Gold dimension",
                "api_policy": "API is built only after Dashboard freeze and consumes Gold only.",
            },
            "contracts": {
                "gold_rule": "select, rename, standardize, merge compatible presentation grains, and aggregate accepted assignments only",
                "no_new_cleaning": True, "no_new_model_fit": True, "no_statistical_retesting": True,
                "no_exact_or_aggregate_coordinates_in_gold": True, "asking_price_not_transaction_price": True,
                "market_temperature_is_listing_market_proxy": True,
                "market_temperature_ranking_universe": "all eligible entities within entity level",
                "four_city_scope_role": "presentation/section-28 attribute only; never used to recompute canonical ranking",
                "segment_fallback_wording": "Market Types / Descriptive Typology",
            },
            "upstream_versions": {name: _source_version(payload) for name, payload in manifests.items()},
            "outputs": {
                "marts": {name: relative_to_project(path) for name, path in outputs.items() if name in EXPECTED_MARTS},
                "dimensions": {name: relative_to_project(path) for name, path in outputs.items() if name in EXPECTED_DIMENSIONS},
                "artifact_contract": relative_to_project(artifact_contract_path),
                "relationship_contract": relative_to_project(relationship_contract_path),
                "page_registry": relative_to_project(page_registry_path),
                "quality_status": relative_to_project(quality_status_path),
                "lineage": relative_to_project(lineage_path), "inventory": relative_to_project(inventory_path),
                "checks": relative_to_project(checks_path),
            },
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        },
        manifest_path,
    )
    show_progress(100, f"complete; status={status['overall_status']}", final=True)
    outputs.update({
        "artifact_contract": artifact_contract_path, "relationship_contract": relationship_contract_path,
        "page_registry": page_registry_path, "quality_status": quality_status_path, "lineage": lineage_path,
        "inventory": inventory_path, "checks": checks_path, "manifest": manifest_path,
    })
    return outputs

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the canonical M4 Gold data layer: 10 marts, 5 dimensions, and section-29 structural contracts."
    )
    parser.add_argument("--strict", action="store_true", help="Return non-zero when Gold is not PASS/ready.")
    parser.add_argument(
        "--write-csv-copies", action="store_true",
        help="Also write CSV inspection copies next to canonical Parquet artifacts.",
    )
    parser.add_argument(
        "--reset-dashboard-metadata", action="store_true",
        help="Remove generated dashboard metadata from the previous architecture before M4-05 regeneration.",
    )
    parser.add_argument(
        "--reset-gold-structure", action="store_true",
        help="Remove stale legacy/unexpected generated Gold table files before rebuilding the canonical 10+5 architecture.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run(
        write_csv_copies=args.write_csv_copies,
        reset_dashboard_metadata=args.reset_dashboard_metadata,
        reset_gold_structure=args.reset_gold_structure,
    )
    manifest = _load_json(outputs["manifest"])
    print("M4 GOLD BUILD COMPLETED")
    print(f"status: {manifest['status']['overall_status']}")
    print(f"gold_data_ready: {manifest['gold_data_ready']}")
    print(f"ready_for_metadata_design: {manifest['ready_for_metadata_design']}")
    print(f"marts: {relative_to_project(MARTS_DIR)}")
    print(f"dimensions: {relative_to_project(DIMENSIONS_DIR)}")
    print(f"artifact_contract: {relative_to_project(outputs['artifact_contract'])}")
    print(f"relationship_contract: {relative_to_project(outputs['relationship_contract'])}")
    print(f"page_registry: {relative_to_project(outputs['page_registry'])}")
    print(f"quality_status: {relative_to_project(outputs['quality_status'])}")
    print(f"checks: {relative_to_project(outputs['checks'])}")
    print(f"manifest: {relative_to_project(outputs['manifest'])}")
    if args.strict and not manifest["gold_data_ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
