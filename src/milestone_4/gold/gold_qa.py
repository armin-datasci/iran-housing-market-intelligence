from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import polars as pl

from src.common.io_utils import atomic_write_csv, atomic_write_json, atomic_write_text
from src.common.paths import relative_to_project
from src.common.validation import Check, checks_frame, make_check, summarize_checks
from src.milestone_4.gold.build_gold import (
    DIMENSIONS_DIR,
    GOLD_ROOT,
    MARTS_DIR,
    METADATA_DIR,
    QA_DIR,
    SOURCES,
    FOUR_CITY_SLUGS,
    build_location_market,
    build_market_monthly,
    build_model_quality,
    build_price_driver_effects,
    build_price_driver_importance,
    build_segment_monthly_mix,
    build_segment_profile,
    build_seller_type,
    build_text_monthly,
    build_text_signals,
    _load_json,
    _source_version,
)
from src.milestone_4.gold.contracts import (
    ALLOWED_DESCRIPTIVE_SEGMENT_METHODS,
    COMMON_MART_METADATA_COLUMNS,
    DIMENSION_KEYS,
    EXPECTED_DIMENSIONS,
    EXPECTED_MARTS,
    EXPECTED_TEXT_SIGNALS,
    FORBIDDEN_COLUMN_TOKENS,
    GOLD_QA_VERSION,
    LEGACY_ARTIFACT_NAMES,
    LEGACY_PAGE_NAMES,
    MART_GRAIN_KEYS,
    RELATIONSHIP_SPECS,
    REQUIRED_ARTIFACT_COLUMNS,
    SEMANTIC_ONLY_DIMENSIONS,
    artifact_contract_rows,
    dashboard_page_rows,
    relationship_contract_rows,
)

VERSION = GOLD_QA_VERSION
PROJECT_ROOT = GOLD_ROOT.parents[1]
REPORT_PATH = PROJECT_ROOT / "reports" / "milestone_4" / "gold_qa" / "Technical_Report_M4_03_Gold_QA.md"
ALLOWED_RELIABILITY_STATUSES = {
    "pass", "ready", "reliable", "validated", "descriptive", "review", "low_n",
    "not_applicable", "fixed_estimate",
}


def _read_csv(path: Path) -> pl.DataFrame:
    return pl.read_csv(path, infer_schema_length=10000, try_parse_dates=False)


def _read_parquet(path: Path) -> pl.DataFrame:
    return pl.read_parquet(path)


def _require_columns(frame: pl.DataFrame, required: Iterable[str], name: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _duplicate_count(frame: pl.DataFrame, keys: list[str]) -> int:
    if frame.height == 0:
        return 0
    _require_columns(frame, keys, "grain check")
    return int(frame.select(keys).is_duplicated().sum())


def _missing_dimension_keys(
    fact: pl.DataFrame, fact_key: str, dimension: pl.DataFrame, dim_key: str
) -> list[str]:
    fact_values = set(fact.get_column(fact_key).drop_nulls().cast(pl.String).to_list())
    dim_values = set(dimension.get_column(dim_key).drop_nulls().cast(pl.String).to_list())
    return sorted(fact_values - dim_values)


def _forbidden_columns(frames: dict[str, pl.DataFrame]) -> list[str]:
    violations: list[str] = []
    for name, frame in frames.items():
        for column in frame.columns:
            if any(token in column.lower() for token in FORBIDDEN_COLUMN_TOKENS):
                violations.append(f"{name}:{column}")
    return sorted(violations)


def _compare_keyed(
    source: pl.DataFrame,
    gold: pl.DataFrame,
    *,
    keys: list[str],
    metric_pairs: list[tuple[str, str, str]],
    tolerance: float = 1e-9,
) -> dict[str, int]:
    _require_columns(source, keys + [left for left, _, _ in metric_pairs], "source reconciliation")
    _require_columns(gold, keys + [right for _, right, _ in metric_pairs], "Gold reconciliation")
    source_select = source.select(
        keys
        + [pl.col(left).alias(f"__src_{idx}") for idx, (left, _, _) in enumerate(metric_pairs)]
        + [pl.lit(1).alias("__src_present")]
    )
    gold_select = gold.select(
        keys
        + [pl.col(right).alias(f"__gold_{idx}") for idx, (_, right, _) in enumerate(metric_pairs)]
        + [pl.lit(1).alias("__gold_present")]
    )
    joined = source_select.join(gold_select, on=keys, how="full", coalesce=True, nulls_equal=True)
    missing_rows = joined.filter(
        pl.col("__src_present").is_null() | pl.col("__gold_present").is_null()
    ).height
    mismatch_rows = 0
    for idx, (_, _, kind) in enumerate(metric_pairs):
        left = pl.col(f"__src_{idx}")
        right = pl.col(f"__gold_{idx}")
        null_mismatch = left.is_null() != right.is_null()
        if kind == "numeric":
            diff = (left.cast(pl.Float64, strict=False) - right.cast(pl.Float64, strict=False)).abs()
            mismatch = null_mismatch | (left.is_not_null() & right.is_not_null() & (diff > tolerance))
        elif kind == "bool":
            mismatch = null_mismatch | (
                left.is_not_null() & right.is_not_null()
                & (left.cast(pl.Boolean, strict=False) != right.cast(pl.Boolean, strict=False))
            )
        else:
            mismatch = null_mismatch | (
                left.is_not_null() & right.is_not_null()
                & (left.cast(pl.String) != right.cast(pl.String))
            )
        mismatch_rows += joined.filter(mismatch).height
    return {"missing_rows": int(missing_rows), "metric_mismatches": int(mismatch_rows)}


def _add_reconciliation(
    checks: list[Check], check_id: str, source: pl.DataFrame, gold: pl.DataFrame,
    *, keys: list[str], metric_pairs: list[tuple[str, str, str]], tolerance: float = 1e-9,
) -> None:
    result = _compare_keyed(source, gold, keys=keys, metric_pairs=metric_pairs, tolerance=tolerance)
    passed = result["missing_rows"] == 0 and result["metric_mismatches"] == 0 and source.height == gold.height
    checks.append(
        make_check(
            check_id, "canonical_reconciliation",
            f"source_rows={source.height};gold_rows={gold.height};missing={result['missing_rows']};mismatches={result['metric_mismatches']}",
            "same row set and zero metric mismatches", passed,
        )
    )


def _load_gold() -> tuple[
    dict[str, pl.DataFrame], dict[str, pl.DataFrame], pl.DataFrame, pl.DataFrame,
    pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, dict[str, Any],
]:
    marts = {name: _read_parquet(MARTS_DIR / f"{name}.parquet") for name in EXPECTED_MARTS}
    dimensions = {name: _read_parquet(DIMENSIONS_DIR / f"{name}.parquet") for name in EXPECTED_DIMENSIONS}
    artifact_contract = _read_csv(METADATA_DIR / "gold_artifact_contract.csv")
    relationship_contract = _read_csv(METADATA_DIR / "gold_relationship_contract.csv")
    page_registry = _read_csv(METADATA_DIR / "dashboard_page_registry.csv")
    quality_status = _read_csv(METADATA_DIR / "dashboard_quality_status.csv")
    lineage = _read_csv(METADATA_DIR / "gold_source_lineage.csv")
    inventory = _read_csv(METADATA_DIR / "gold_artifact_inventory.csv")
    manifest = _load_json(QA_DIR / "gold_manifest.json")
    return (
        marts, dimensions, artifact_contract, relationship_contract, page_registry,
        quality_status, lineage, inventory, manifest,
    )


def _artifact_completeness_checks(checks: list[Check]) -> None:
    expected_marts = {f"{name}.parquet" for name in EXPECTED_MARTS}
    observed_marts = {path.name for path in MARTS_DIR.glob("*.parquet")}
    checks.append(make_check("mart_file_set_exact", "architecture", sorted(observed_marts), sorted(expected_marts), observed_marts == expected_marts))
    expected_dims = {f"{name}.parquet" for name in EXPECTED_DIMENSIONS}
    observed_dims = {path.name for path in DIMENSIONS_DIR.glob("*.parquet")}
    checks.append(make_check("dimension_file_set_exact", "architecture", sorted(observed_dims), sorted(expected_dims), observed_dims == expected_dims))
    required_metadata = {
        "gold_artifact_contract.csv", "gold_relationship_contract.csv", "dashboard_page_registry.csv",
        "dashboard_quality_status.csv", "gold_source_lineage.csv", "gold_artifact_inventory.csv",
    }
    observed_metadata = {path.name for path in METADATA_DIR.glob("*") if path.is_file()}
    checks.append(make_check("structural_metadata_complete", "architecture", sorted(required_metadata & observed_metadata), sorted(required_metadata), required_metadata.issubset(observed_metadata)))


def _schema_and_grain_checks(
    checks: list[Check], marts: dict[str, pl.DataFrame], dimensions: dict[str, pl.DataFrame]
) -> None:
    for name, frame in marts.items():
        missing = sorted(set(REQUIRED_ARTIFACT_COLUMNS[name] + COMMON_MART_METADATA_COLUMNS) - set(frame.columns))
        checks.append(make_check(f"{name}_required_columns", "schema", missing, [], not missing))
        grain = list(MART_GRAIN_KEYS[name])
        duplicates = _duplicate_count(frame, grain)
        grain_nulls = sum(frame.get_column(column).null_count() for column in grain)
        checks.append(make_check(f"{name}_grain_unique", "grain", duplicates, 0, duplicates == 0))
        checks.append(make_check(f"{name}_grain_nonnull", "grain", grain_nulls, 0, grain_nulls == 0))
        statuses = set(value.lower() for value in frame.get_column("reliability_status").drop_nulls().cast(pl.String).to_list())
        invalid = sorted(statuses - ALLOWED_RELIABILITY_STATUSES)
        checks.append(make_check(f"{name}_reliability_status_values", "semantics", invalid, [], not invalid))
    for name, frame in dimensions.items():
        missing = sorted(set(REQUIRED_ARTIFACT_COLUMNS[name]) - set(frame.columns))
        checks.append(make_check(f"{name}_required_columns", "schema", missing, [], not missing))
        key = DIMENSION_KEYS[name]
        duplicates = _duplicate_count(frame, [key])
        nulls = frame.get_column(key).null_count()
        checks.append(make_check(f"{name}_unique_key", "dimension_integrity", duplicates, 0, duplicates == 0))
        checks.append(make_check(f"{name}_nonnull_key", "dimension_integrity", nulls, 0, nulls == 0))


def _referential_integrity_checks(
    checks: list[Check], marts: dict[str, pl.DataFrame], dimensions: dict[str, pl.DataFrame]
) -> None:
    for spec in RELATIONSHIP_SPECS:
        fact = marts[spec.mart_name]
        dimension = dimensions[spec.dimension_name]
        missing = _missing_dimension_keys(fact, spec.fact_key, dimension, spec.dimension_key)
        checks.append(make_check(f"fk_{spec.relationship_id}", "referential_integrity", missing[:20], [], not missing, notes=f"Missing key count={len(missing)}"))
        if not spec.nullable_fact_key:
            nulls = fact.get_column(spec.fact_key).null_count()
            checks.append(make_check(f"fk_{spec.relationship_id}_nonnull", "referential_integrity", nulls, 0, nulls == 0))
    seller = marts["mart_seller_type"]
    dim = dimensions["dim_user_type"]
    seller_values: set[str] = set()
    for column in ["agency_label", "personal_label"]:
        seller_values.update(seller.get_column(column).drop_nulls().cast(pl.String).to_list())
    dim_values = set(dim.get_column("user_type").drop_nulls().cast(pl.String).to_list())
    checks.append(make_check("semantic_user_type_values_complete", "referential_integrity", sorted(seller_values - dim_values), [], seller_values.issubset(dim_values)))
    checks.append(make_check("semantic_only_dimension_contract", "architecture", sorted(SEMANTIC_ONLY_DIMENSIONS), ["dim_user_type"], SEMANTIC_ONLY_DIMENSIONS == {"dim_user_type"}))


def _contract_checks(
    checks: list[Check], artifact_contract: pl.DataFrame,
    relationship_contract: pl.DataFrame, page_registry: pl.DataFrame,
) -> None:
    expected_artifacts = pl.DataFrame(artifact_contract_rows()).sort(["artifact_type", "artifact_name"])
    observed_artifacts = artifact_contract.sort(["artifact_type", "artifact_name"]).select(expected_artifacts.columns)
    checks.append(make_check("artifact_contract_matches_code", "contract", observed_artifacts.height, expected_artifacts.height, observed_artifacts.shape == expected_artifacts.shape and observed_artifacts.to_dicts() == expected_artifacts.to_dicts()))

    expected_relationships = pl.DataFrame(relationship_contract_rows()).sort("relationship_id")
    observed_relationships = relationship_contract.sort("relationship_id").select(expected_relationships.columns)
    checks.append(make_check("relationship_contract_matches_code", "contract", observed_relationships.height, expected_relationships.height, observed_relationships.shape == expected_relationships.shape and observed_relationships.to_dicts() == expected_relationships.to_dicts()))
    invalid_relationships = relationship_contract.filter(
        (pl.col("relationship_type") != "physical_one_to_many")
        | (pl.col("cardinality") != "1:*")
        | (pl.col("cross_filter_direction") != "Single")
        | (pl.col("active") != True)
        | (pl.col("expected_unique_one_side") != True)
    ).height
    checks.append(make_check("relationship_contract_star_schema_rules", "contract", invalid_relationships, 0, invalid_relationships == 0))
    checks.append(make_check("no_fake_disconnected_relationship_rows", "contract", relationship_contract.filter(pl.col("active") != True).height, 0, relationship_contract.filter(pl.col("active") != True).height == 0))

    expected_pages = dashboard_page_rows()
    observed_pages = page_registry.sort("page_order").select(["page_id", "page_order", "page_name"]).to_dicts()
    checks.append(make_check("page_registry_exact_section_29", "dashboard_pages", observed_pages, expected_pages, observed_pages == expected_pages))
    legacy_pages = sorted(set(page_registry.get_column("page_name").cast(pl.String).to_list()) & LEGACY_PAGE_NAMES)
    checks.append(make_check("page_registry_has_no_legacy_names", "dashboard_pages", legacy_pages, [], not legacy_pages))


def _semantic_checks(
    checks: list[Check], marts: dict[str, pl.DataFrame], dimensions: dict[str, pl.DataFrame],
    quality_status: pl.DataFrame, inventory: pl.DataFrame,
) -> None:
    forbidden = _forbidden_columns({**marts, **dimensions})
    checks.append(make_check("no_coordinates_or_legacy_columns", "privacy", forbidden, [], not forbidden))
    observed_inventory = set(inventory.get_column("artifact_name").cast(pl.String).to_list())
    legacy_inventory = sorted(observed_inventory & LEGACY_ARTIFACT_NAMES)
    checks.append(make_check("inventory_has_no_legacy_artifacts", "legacy", legacy_inventory, [], not legacy_inventory))

    signals = set(marts["mart_text_signals"].get_column("keyword_family").cast(pl.String).to_list())
    monthly_signals = set(marts["mart_text_monthly"].get_column("keyword_family").cast(pl.String).to_list())
    checks.append(make_check("text_signals_exact_six", "text", sorted(signals), sorted(EXPECTED_TEXT_SIGNALS), signals == set(EXPECTED_TEXT_SIGNALS)))
    checks.append(make_check("text_monthly_exact_six", "text", sorted(monthly_signals), sorted(EXPECTED_TEXT_SIGNALS), monthly_signals == set(EXPECTED_TEXT_SIGNALS)))

    importance_ids = set(
        marts["mart_price_driver_importance"].get_column("feature_or_block_id")
        .drop_nulls().cast(pl.String).to_list()
    )
    grouped_blocks_ok = {"location", "property_type"}.issubset(importance_ids)
    checks.append(make_check(
        "price_driver_grouped_control_blocks_present", "price_drivers",
        sorted({"location", "property_type"} - importance_ids), [], grouped_blocks_ok,
        notes="P05 predictive contribution must retain grouped City+Neighborhood and Property-family+Category controls."
    ))

    methods = set(marts["mart_segment_profile"].get_column("method_type").drop_nulls().cast(pl.String).to_list())
    descriptive_ok = bool(methods) and methods.issubset(ALLOWED_DESCRIPTIVE_SEGMENT_METHODS)
    checks.append(make_check("segment_methods_descriptive_and_named", "segments", sorted(methods), sorted(ALLOWED_DESCRIPTIVE_SEGMENT_METHODS), descriptive_ok))
    segment_count = marts["mart_segment_profile"].get_column("segment_id").n_unique()
    checks.append(make_check("segment_count_interpretable_cap", "segments", segment_count, "1..5", 1 <= segment_count <= 5))

    location = marts["mart_location_market"]
    temp = location.filter(pl.col("temperature_available_flag") == True)
    temp_rank_nulls = temp.get_column("market_temperature_rank").null_count()
    checks.append(make_check("location_temperature_rank_present", "temperature", temp_rank_nulls, 0, temp_rank_nulls == 0))
    four_city_dependency = temp.filter(
        (pl.col("entity_level") == "neighborhood")
        & (pl.col("temperature_reliability_eligible_flag") == True)
        & (pl.col("professor_facing_eligible_flag") != True)
    ).height
    checks.append(make_check(
        "temperature_surface_gate_is_all_city_not_four_city", "temperature",
        four_city_dependency, 0, four_city_dependency == 0,
        notes="Professor-facing eligibility in Gold is the all-city reliability gate; four-city coverage is a location attribute/§28 concern.",
    ))
    price_map = location.filter(pl.col("price_map_available_flag") == True)
    outside_four_price = price_map.filter(~pl.col("city_slug").is_in(list(FOUR_CITY_SLUGS))).height
    checks.append(make_check(
        "price_map_population_is_not_four_city_only", "scope",
        outside_four_price, ">0", outside_four_price > 0,
        notes="Canonical M3 Market Map / Gold Price Map must cover all eligible cities; the four-city view is not an upstream population filter.",
    ))
    outside_four_temp = temp.filter(~pl.col("city_slug").is_in(list(FOUR_CITY_SLUGS))).height
    checks.append(make_check(
        "temperature_population_is_not_four_city_only", "scope",
        outside_four_temp, ">0", outside_four_temp > 0,
        notes="Canonical temperature ranking remains all-city; four-city display is presentation-only.",
    ))
    reliable_price = price_map.filter(pl.col("reliable_flag") == True)
    missing_price_ranks = reliable_price.filter(
        pl.col("affordable_rank").is_null() | pl.col("expensive_rank").is_null()
    ).height
    checks.append(make_check(
        "reliable_price_map_has_canonical_ranks", "price_map",
        missing_price_ranks, 0, missing_price_ranks == 0,
        notes="P03 affordable/expensive views use static ranks over already-published reliable neighborhood estimates.",
    ))

    _require_columns(quality_status, [
        "stage", "component", "overall_status", "ready", "critical_failures", "review_count",
        "reliability_status", "sample_n", "source_manifest",
    ], "dashboard_quality_status.csv")
    stages = set(quality_status.get_column("stage").cast(pl.String).to_list())
    checks.append(make_check("quality_status_has_m1_to_m4", "data_quality_page", sorted(stages), ["M1", "M2", "M3", "M4"], {"M1", "M2", "M3", "M4"}.issubset(stages)))
    missing_core = quality_status.filter(pl.col("stage").is_in(["M1", "M2"]) & (pl.col("overall_status") == "MISSING")).height
    checks.append(make_check(
        "quality_status_core_upstream_present", "data_quality_page",
        missing_core, 0, missing_core == 0,
        notes="P02 Data Quality requires the canonical M1 and M2 closeout/status artifacts."
    ))
    core_not_ready = quality_status.filter(
        pl.col("stage").is_in(["M1", "M2"])
        & (
            (pl.col("ready") != True)
            | (pl.col("critical_failures").cast(pl.Int64, strict=False).fill_null(1) > 0)
        )
    ).height
    checks.append(make_check(
        "quality_status_core_upstream_ready", "data_quality_page",
        core_not_ready, 0, core_not_ready == 0,
        notes=(
            "M1/M2 readiness is distinct from REVIEW status: documented non-critical "
            "reviews may remain visible, but critical failures or ready=false block Gold."
        ),
    ))


def _reconciliation_checks(checks: list[Check], marts: dict[str, pl.DataFrame], build_manifest: dict[str, Any]) -> None:
    versions = build_manifest.get("upstream_versions", {})
    expected_monthly = build_market_monthly(str(versions.get("monthly", "unknown")))
    gold_monthly = marts["mart_market_monthly"].select(expected_monthly.columns)
    _add_reconciliation(
        checks, "reconcile_market_monthly", expected_monthly, gold_monthly,
        keys=["analysis_month", "entity_level", "location_key", "market_scope"],
        metric_pairs=[
            ("deduplicated_listing_count", "deduplicated_listing_count", "numeric"),
            ("median_asking_price_per_sqm_toman", "median_asking_price_per_sqm_toman", "numeric"),
            ("median_price_mom_pct", "median_price_mom_pct", "numeric"),
        ],
    )

    market_map_manifest = _load_json(SOURCES["market_map_manifest"])
    temp_manifest = _load_json(SOURCES["market_temperature_manifest"])
    expected_location = build_location_market(market_map_manifest, temp_manifest)
    gold_location = marts["mart_location_market"].select(expected_location.columns)
    _add_reconciliation(
        checks, "reconcile_location_market", expected_location, gold_location,
        keys=["entity_level", "location_key"],
        metric_pairs=[
            ("median_asking_price_per_sqm_toman", "median_asking_price_per_sqm_toman", "numeric"),
            ("reliable_flag", "reliable_flag", "bool"),
            ("market_temperature_score", "market_temperature_score", "numeric"),
            ("market_temperature_rank", "market_temperature_rank", "numeric"),
            ("temperature_reliability_eligible_flag", "temperature_reliability_eligible_flag", "bool"),
        ],
    )

    price_version = str(versions.get("price_drivers", "unknown"))
    for name, builder, key, pairs in [
        ("mart_price_driver_effects", build_price_driver_effects, ["feature_id"], [("adjusted_effect_pct", "adjusted_effect_pct", "numeric"), ("ci_lower_pct", "ci_lower_pct", "numeric"), ("ci_upper_pct", "ci_upper_pct", "numeric")]),
        ("mart_price_driver_importance", build_price_driver_importance, ["feature_or_block_id"], [("permutation_importance", "permutation_importance", "numeric"), ("permutation_importance_sd", "permutation_importance_sd", "numeric")]),
    ]:
        expected = builder(price_version)
        gold = marts[name].select(expected.columns)
        _add_reconciliation(checks, f"reconcile_{name}", expected, gold, keys=key, metric_pairs=pairs)

    expected_model = build_model_quality(price_version)
    # standardize_mart adds the composite key; compare on the natural fields instead.
    natural_keys = ["record_type", "model_name", "population", "evaluation_split", "evaluation_stage", "error_scope", "city_slug", "property_family", "benchmark_role"]
    gold_model = marts["mart_model_quality"].select(expected_model.columns)
    _add_reconciliation(
        checks, "reconcile_model_quality", expected_model, gold_model, keys=natural_keys,
        metric_pairs=[("n", "n", "numeric"), ("rmse_log", "rmse_log", "numeric"), ("r2_log", "r2_log", "numeric"), ("median_ape_pct", "median_ape_pct", "numeric"), ("p90_ape_pct", "p90_ape_pct", "numeric")],
    )

    seller_version = str(versions.get("seller", "unknown"))
    expected_seller = build_seller_type(seller_version)
    gold_seller = marts["mart_seller_type"].select(expected_seller.columns)
    _add_reconciliation(checks, "reconcile_seller_type", expected_seller, gold_seller, keys=["comparison"], metric_pairs=[("agency_n", "agency_n", "numeric"), ("personal_n", "personal_n", "numeric"), ("adjusted_crossfit_difference_pct", "adjusted_crossfit_difference_pct", "numeric"), ("adjusted_ci_low_pct", "adjusted_ci_low_pct", "numeric"), ("adjusted_ci_high_pct", "adjusted_ci_high_pct", "numeric")])

    text_version = str(versions.get("text", "unknown"))
    for name, builder, keys, pairs in [
        ("mart_text_signals", build_text_signals, ["keyword_family"], [("manual_precision", "manual_precision", "numeric"), ("adjusted_effect_pct", "adjusted_effect_pct", "numeric"), ("q_value", "q_value", "numeric")]),
        ("mart_text_monthly", build_text_monthly, ["month_key", "keyword_family"], [("sale_valid_listing_count", "sale_valid_listing_count", "numeric"), ("matched_listing_count", "matched_listing_count", "numeric"), ("matched_rate", "matched_rate", "numeric")]),
    ]:
        expected = builder(text_version)
        gold = marts[name].select(expected.columns)
        _add_reconciliation(checks, f"reconcile_{name}", expected, gold, keys=keys, metric_pairs=pairs)

    segment_manifest = _load_json(SOURCES["segment_manifest"])
    expected_profile = build_segment_profile(segment_manifest)
    gold_profile = marts["mart_segment_profile"].select(expected_profile.columns)
    _add_reconciliation(checks, "reconcile_segment_profile", expected_profile, gold_profile, keys=["segment_id"], metric_pairs=[("listing_n", "listing_n", "numeric"), ("listing_share_pct", "listing_share_pct", "numeric"), ("median_asking_price_per_sqm_toman", "median_asking_price_per_sqm_toman", "numeric"), ("method_type", "method_type", "string")])
    expected_mix = build_segment_monthly_mix(segment_manifest)
    gold_mix = marts["mart_segment_monthly_mix"].select(expected_mix.columns)
    _add_reconciliation(checks, "reconcile_segment_monthly_mix", expected_mix, gold_mix, keys=["month_key", "segment_id"], metric_pairs=[("listing_n", "listing_n", "numeric"), ("month_total_n", "month_total_n", "numeric"), ("listing_share_pct", "listing_share_pct", "numeric")])


def _append_qa_quality_row(quality_status: pl.DataFrame, status: dict[str, Any], manifest_path: Path) -> pl.DataFrame:
    row = pl.DataFrame([
        {
            "stage": "M4", "component": "Gold QA", "overall_status": str(status.get("overall_status", "UNKNOWN")).upper(),
            "ready": bool(status.get("ready") and str(status.get("overall_status", "")).upper() == "PASS"),
            "critical_failures": status.get("critical_failures", 0), "review_count": status.get("review_count", status.get("warnings", 0)),
            "reliability_status": "ready" if bool(status.get("ready") and str(status.get("overall_status", "")).upper() == "PASS") else "review",
            "sample_n": None, "source_manifest": relative_to_project(manifest_path),
        }
    ])
    return pl.concat([quality_status.filter(pl.col("component") != "Gold QA"), row], how="diagonal_relaxed")


def _build_report(status: dict[str, Any], checks: list[Check]) -> str:
    failed = [check for check in checks if str(check.status).upper() == "FAIL"]
    review = [check for check in checks if str(check.status).upper() == "REVIEW"]
    ready = bool(status.get("ready") and str(status.get("overall_status", "")).upper() == "PASS")
    lines = [
        "# Technical Report - M4 Gold Data Contract QA", "",
        f"- Version: `{VERSION}`", f"- Overall status: `{status.get('overall_status')}`",
        f"- Critical failures: `{status.get('critical_failures')}`", f"- Review checks: `{len(review)}`",
        f"- Ready for M4-05 Measure/DAX design: `{ready}`", "",
        "## Frozen architecture", "",
        "- 10 dashboard-facing marts and 5 conformed dimensions.",
        "- Exact section-29 P01-P08 page registry.",
        "- `mart_location_market` combines compatible Price Map and Market Temperature location grains.",
        "- `mart_text_monthly` supplies month-sensitive text frequency; adjusted text estimates remain fixed in `mart_text_signals`.",
        "- Segment profile and monthly mix carry segment ID/name directly; no `dim_segment` is persisted.",
        "- `dim_user_type` is semantic-only; no fake inactive/bidirectional relationship is created.", "",
        "## Interpretation constraints", "",
        "- Asking prices are not verified transaction prices.",
        "- Listing activity is not physical inventory, liquidity, or absorption.",
        "- Market Temperature uses the frozen all-city ranking; four-city scope is presentation/§28 only.",
        "- Driver, seller, and text effects are observational/fixed estimates, not causal effects.",
        "- Segment fallback is presented as Market Types / Descriptive Typology.",
        "- Exact coordinates are excluded from Gold.", "",
    ]
    if failed:
        lines += ["## Failed checks", ""]
        for check in failed:
            lines.append(f"- `{check.check_id}`: actual `{check.actual}`, expected `{check.expected}`.")
        lines.append("")
    return "\n".join(lines)


def run() -> dict[str, Path]:
    checks: list[Check] = []
    _artifact_completeness_checks(checks)
    (
        marts, dimensions, artifact_contract, relationship_contract, page_registry,
        quality_status, lineage, inventory, build_manifest,
    ) = _load_gold()

    checks.append(make_check("gold_build_ready", "build_qa", build_manifest.get("gold_data_ready", False), True, bool(build_manifest.get("gold_data_ready", False))))
    architecture = build_manifest.get("architecture", {})
    checks.append(make_check("manifest_10_marts", "architecture", architecture.get("gold_marts"), 10, architecture.get("gold_marts") == 10))
    checks.append(make_check("manifest_5_dimensions", "architecture", architecture.get("conformed_dimensions"), 5, architecture.get("conformed_dimensions") == 5))
    checks.append(make_check("manifest_semantic_only_user_type", "architecture", architecture.get("semantic_only_dimensions"), ["dim_user_type"], architecture.get("semantic_only_dimensions") == ["dim_user_type"]))

    _schema_and_grain_checks(checks, marts, dimensions)
    _referential_integrity_checks(checks, marts, dimensions)
    _contract_checks(checks, artifact_contract, relationship_contract, page_registry)
    _semantic_checks(checks, marts, dimensions, quality_status, inventory)
    _reconciliation_checks(checks, marts, build_manifest)

    build_checks = _read_csv(QA_DIR / "gold_checks.csv")
    build_failures = build_checks.filter(pl.col("status").cast(pl.String).str.to_uppercase() == "FAIL").height
    checks.append(make_check("gold_build_checks_zero_fail", "build_qa", build_failures, 0, build_failures == 0))

    status = summarize_checks(checks)
    gold_data_ready = bool(status.get("ready") and str(status.get("overall_status", "")).upper() == "PASS")
    checks_path = QA_DIR / "gold_qa_checks.csv"
    manifest_path = QA_DIR / "gold_qa_manifest.json"
    quality_status_path = METADATA_DIR / "dashboard_quality_status.csv"
    atomic_write_csv(checks_frame(checks), checks_path)
    atomic_write_json(
        {
            "version": VERSION, "status": status, "gold_data_contract_ready": gold_data_ready,
            "ready_for_measure_dax_design": gold_data_ready,
            "dashboard_metadata_status": "PENDING_M4_05_GENERATION",
            "architecture": {
                "dashboard_pages": dashboard_page_rows(), "gold_marts": len(EXPECTED_MARTS),
                "conformed_dimensions": len(EXPECTED_DIMENSIONS), "physical_relationships": len(RELATIONSHIP_SPECS),
                "semantic_only_dimensions": sorted(SEMANTIC_ONLY_DIMENSIONS),
            },
            "qa_scope": [
                "artifact_completeness", "schema_and_grain", "dimension_integrity", "referential_integrity",
                "source_reconciliation", "structural_contracts", "section_29_pages", "semantic_integrity",
                "data_quality_page_status", "legacy_contamination", "spatial_privacy",
            ],
            "outputs": {"checks": relative_to_project(checks_path), "technical_report": relative_to_project(REPORT_PATH), "quality_status": relative_to_project(quality_status_path)},
        },
        manifest_path,
    )
    updated_quality = _append_qa_quality_row(quality_status, status, manifest_path)
    atomic_write_csv(updated_quality, quality_status_path)
    inventory_path = METADATA_DIR / "gold_artifact_inventory.csv"
    updated_inventory = inventory.with_columns(
        pl.when(pl.col("artifact_name") == "dashboard_quality_status")
        .then(pl.lit(updated_quality.height))
        .otherwise(pl.col("row_count"))
        .alias("row_count")
    )
    atomic_write_csv(updated_inventory, inventory_path)
    atomic_write_text(_build_report(status, checks), REPORT_PATH)

    print("M4 GOLD QA COMPLETED")
    print(f"overall_status: {status.get('overall_status')}")
    print(f"critical_failures: {status.get('critical_failures')}")
    print(f"reviews: {status.get('review_count', status.get('warnings', 0))}")
    print(f"gold_data_contract_ready: {gold_data_ready}")
    print(f"ready_for_measure_dax_design: {gold_data_ready}")
    print(f"checks: {relative_to_project(checks_path)}")
    print(f"manifest: {relative_to_project(manifest_path)}")
    print(f"quality_status: {relative_to_project(quality_status_path)}")
    print(f"report: {relative_to_project(REPORT_PATH)}")
    return {"checks": checks_path, "manifest": manifest_path, "quality_status": quality_status_path, "report": REPORT_PATH}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate canonical M4 Gold before Measure/DAX contract generation.")
    parser.add_argument("--strict", action="store_true", help="Return non-zero when Gold QA is not PASS/ready.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run()
    if args.strict:
        manifest = _load_json(outputs["manifest"])
        if not manifest.get("gold_data_contract_ready", False):
            raise SystemExit(2)


if __name__ == "__main__":
    main()
