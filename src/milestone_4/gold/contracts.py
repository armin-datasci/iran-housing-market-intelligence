"""Canonical structural contracts for Milestone 4 Gold.

This module is the single source of truth for the dashboard-facing Gold
architecture. It intentionally contains only structural contracts; Measure/DAX
contracts are generated later in M4-05 after Gold QA passes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

GOLD_BUILD_VERSION: Final = "m4-gold-v3.4-section29-status-location-handoff-hotfix"
GOLD_QA_VERSION: Final = "m4-gold-qa-v3.4-section29-status-location-handoff-hotfix"


@dataclass(frozen=True, slots=True)
class DashboardPage:
    page_id: str
    page_order: int
    page_name: str


DASHBOARD_PAGE_SPECS: Final[tuple[DashboardPage, ...]] = (
    DashboardPage("P01", 1, "Executive Market Overview"),
    DashboardPage("P02", 2, "Data Quality"),
    DashboardPage("P03", 3, "Price Map"),
    DashboardPage("P04", 4, "Supply and Price Trends"),
    DashboardPage("P05", 5, "Amenities and Price Drivers"),
    DashboardPage("P06", 6, "Seller Type Comparison"),
    DashboardPage("P07", 7, "Text Signals"),
    DashboardPage("P08", 8, "Market Segments"),
)
DASHBOARD_PAGES: Final[dict[str, str]] = {
    page.page_id: page.page_name for page in DASHBOARD_PAGE_SPECS
}

MANDATORY_FILTER_IDS: Final[tuple[str, ...]] = (
    "city",
    "neighborhood",
    "property",
    "month",
    "price_regime",
    "user_type",
    "minimum_valid_listings",
)

EXPECTED_TEXT_SIGNALS: Final[frozenset[str]] = frozenset(
    {"new_build", "unused", "urgent", "exchange", "below_market", "migration_sale"}
)

# Minimal dashboard-facing Gold: every table has a distinct semantic/grain role.
EXPECTED_MARTS: Final[tuple[str, ...]] = (
    "mart_market_monthly",
    "mart_location_market",
    "mart_price_driver_effects",
    "mart_price_driver_importance",
    "mart_model_quality",
    "mart_seller_type",
    "mart_text_signals",
    "mart_text_monthly",
    "mart_segment_profile",
    "mart_segment_monthly_mix",
)

EXPECTED_DIMENSIONS: Final[tuple[str, ...]] = (
    "dim_location",
    "dim_month",
    "dim_property",
    "dim_price_regime",
    "dim_user_type",
)

DIMENSION_KEYS: Final[dict[str, str]] = {
    "dim_location": "location_key",
    "dim_month": "month_key",
    "dim_property": "property_key",
    "dim_price_regime": "price_regime_key",
    "dim_user_type": "user_type_key",
}

# dim_user_type is intentionally semantic-only because the seller comparison is a
# fixed contrast row. Later DAX uses the user-type selection for descriptive output
# without creating a misleading physical fact relationship.
SEMANTIC_ONLY_DIMENSIONS: Final[frozenset[str]] = frozenset({"dim_user_type"})
# Backward-compatible import name for older callers/tests; no pseudo-relationship
# row is generated for it.
DISCONNECTED_DIMENSIONS: Final[frozenset[str]] = SEMANTIC_ONLY_DIMENSIONS

MART_GRAIN_KEYS: Final[dict[str, tuple[str, ...]]] = {
    "mart_market_monthly": ("analysis_month", "entity_level", "location_key", "market_scope"),
    "mart_location_market": ("entity_level", "location_key"),
    "mart_price_driver_effects": ("feature_id",),
    "mart_price_driver_importance": ("feature_or_block_id",),
    "mart_model_quality": ("model_quality_key",),
    "mart_seller_type": ("comparison",),
    "mart_text_signals": ("keyword_family",),
    "mart_text_monthly": ("month_key", "keyword_family"),
    "mart_segment_profile": ("segment_id",),
    "mart_segment_monthly_mix": ("month_key", "segment_id"),
}

COMMON_MART_METADATA_COLUMNS: Final[tuple[str, ...]] = (
    "source_task_id",
    "source_artifact",
    "population_definition",
    "reliability_status",
    "reliability_reason",
    "limitation_note",
    "method_version",
    "sample_n",
    "price_observation_type",
    "price_unit",
)

REQUIRED_ARTIFACT_COLUMNS: Final[dict[str, tuple[str, ...]]] = {
    "mart_market_monthly": (
        "analysis_month", "month_key", "entity_level", "location_key", "market_scope",
        "series_kind", "deduplicated_listing_count",
    ),
    "mart_location_market": (
        "entity_level", "location_key", "city_slug", "property_key", "price_regime_key",
        "price_map_available_flag", "temperature_available_flag",
        "median_asking_price_per_sqm_toman", "reliable_flag", "affordable_rank", "expensive_rank",
        "market_temperature_score", "market_temperature_rank",
        "listing_activity_trend_pct_per_month", "temperature_reliability_eligible_flag",
    ),
    "mart_price_driver_effects": (
        "feature_id", "feature_display_name", "adjusted_effect_pct", "ci_lower_pct", "ci_upper_pct",
    ),
    "mart_price_driver_importance": (
        "feature_or_block_id", "feature_or_block_name", "permutation_importance",
        "permutation_importance_sd",
    ),
    "mart_model_quality": (
        "model_quality_key", "record_type", "model_name", "population", "evaluation_split",
        "n", "rmse_log", "r2_log", "median_ape_pct", "p75_ape_pct", "p90_ape_pct",
        "within_20_pct", "within_30_pct", "within_50_pct", "median_absolute_error_psm_toman",
    ),
    "mart_seller_type": (
        "comparison", "agency_label", "personal_label", "agency_n", "personal_n",
        "adjusted_crossfit_difference_pct", "adjusted_ci_low_pct", "adjusted_ci_high_pct",
    ),
    "mart_text_signals": (
        "keyword_family", "keyword_display_name", "manual_precision", "adjusted_effect_pct", "q_value",
    ),
    "mart_text_monthly": (
        "analysis_month", "month_key", "keyword_family", "keyword_display_name",
        "sale_valid_listing_count", "matched_listing_count", "matched_rate",
    ),
    "mart_segment_profile": (
        "segment_id", "segment_name", "method_type", "property_key", "listing_n",
        "positioning_listing_n", "median_normalized_price_index",
    ),
    "mart_segment_monthly_mix": (
        "analysis_month", "month_key", "segment_id", "segment_name", "listing_n", "listing_share_pct",
    ),
    "dim_location": ("location_key", "location_level", "location_display_name"),
    "dim_month": ("month_key", "analysis_month", "month_start", "chronological_sort"),
    "dim_property": (
        "property_key", "property_family", "property_display_name", "property_level", "is_observed_in_gold",
    ),
    "dim_price_regime": (
        "price_regime_key", "price_regime", "price_regime_display_name", "is_observed_in_gold",
    ),
    "dim_user_type": ("user_type_key", "user_type", "user_type_display_name"),
}


@dataclass(frozen=True, slots=True)
class RelationshipSpec:
    relationship_id: str
    mart_name: str
    fact_key: str
    dimension_name: str
    dimension_key: str
    nullable_fact_key: bool = False


# Physical Power BI relationships only. All are active, single-direction,
# fact-many -> dimension-one relationships. dim_user_type is semantic-only.
RELATIONSHIP_SPECS: Final[tuple[RelationshipSpec, ...]] = (
    RelationshipSpec("rel_market_monthly_location", "mart_market_monthly", "location_key", "dim_location", "location_key"),
    RelationshipSpec("rel_market_monthly_month", "mart_market_monthly", "month_key", "dim_month", "month_key"),
    RelationshipSpec("rel_market_monthly_property", "mart_market_monthly", "property_key", "dim_property", "property_key", True),
    RelationshipSpec("rel_market_monthly_regime", "mart_market_monthly", "price_regime_key", "dim_price_regime", "price_regime_key", True),
    RelationshipSpec("rel_location_market_location", "mart_location_market", "location_key", "dim_location", "location_key"),
    RelationshipSpec("rel_location_market_property", "mart_location_market", "property_key", "dim_property", "property_key"),
    RelationshipSpec("rel_location_market_regime", "mart_location_market", "price_regime_key", "dim_price_regime", "price_regime_key"),
    RelationshipSpec("rel_seller_regime", "mart_seller_type", "price_regime_key", "dim_price_regime", "price_regime_key"),
    RelationshipSpec("rel_text_signals_regime", "mart_text_signals", "price_regime_key", "dim_price_regime", "price_regime_key"),
    RelationshipSpec("rel_text_monthly_month", "mart_text_monthly", "month_key", "dim_month", "month_key"),
    RelationshipSpec("rel_text_monthly_regime", "mart_text_monthly", "price_regime_key", "dim_price_regime", "price_regime_key"),
    RelationshipSpec("rel_segment_profile_property", "mart_segment_profile", "property_key", "dim_property", "property_key"),
    RelationshipSpec("rel_segment_mix_month", "mart_segment_monthly_mix", "month_key", "dim_month", "month_key"),
)

ALLOWED_DESCRIPTIVE_SEGMENT_METHODS: Final[frozenset[str]] = frozenset(
    {"rule_based_descriptive_typology", "compatible_domain_segment"}
)

FORBIDDEN_COLUMN_TOKENS: Final[tuple[str, ...]] = (
    "latitude", "longitude", "coordinate", "centroid", "geometry", "supply_tightness", "legacy_",
)

# Includes both truly old names and the superseded v2.2 split Location/Temperature
# architecture so --reset-gold-structure leaves exactly the canonical 10+5 set.
LEGACY_ARTIFACT_NAMES: Final[frozenset[str]] = frozenset(
    {
        "dim_market_segment", "dim_segment",
        "mart_city_month", "mart_market_heat", "mart_market_price", "mart_neighborhood_price",
        "mart_pipeline_status", "mart_property_drivers", "mart_quality_risks", "mart_quality_summary",
        "mart_seller_comparison", "mart_supply_by_user_month", "mart_text_frequency_monthly",
        "mart_market_segments", "mart_neighborhood_market", "mart_market_temperature",
    }
)

LEGACY_PAGE_NAMES: Final[frozenset[str]] = frozenset(
    {
        "Price & Supply Trends", "Neighborhood & Spatial Market", "Hot & Cold Markets",
        "Price Drivers & Model Quality", "Seller Type Analysis", "Validated Text Signals",
    }
)

# These files belong to M4-05 and may be cleared before regeneration. Structural
# M4 Gold metadata is deliberately not in this list.
DASHBOARD_METADATA_FILENAMES: Final[tuple[str, ...]] = (
    "dashboard_metric_contract.csv",
    "dashboard_measure_catalog.csv",
    "dashboard_page_contract.csv",
    "dashboard_filter_contract.csv",
    "dashboard_relationship_contract.csv",
)


def dashboard_page_rows() -> list[dict[str, object]]:
    return [
        {"page_id": page.page_id, "page_order": page.page_order, "page_name": page.page_name}
        for page in DASHBOARD_PAGE_SPECS
    ]


def relationship_contract_rows() -> list[dict[str, object]]:
    """Return Power BI relationship rows in the visible 1:* direction.

    Internal specs retain fact->dimension keys for referential-integrity QA, while
    this machine-readable contract is expressed exactly as Power BI should be built:
    dimension (1) -> mart (*) / Single / Active.
    """
    return [
        {
            "relationship_id": spec.relationship_id,
            "relationship_type": "physical_one_to_many",
            "from_table": spec.dimension_name,
            "from_column": spec.dimension_key,
            "to_table": spec.mart_name,
            "to_column": spec.fact_key,
            "cardinality": "1:*",
            "cross_filter_direction": "Single",
            "active": True,
            "nullable_fact_key": spec.nullable_fact_key,
            "expected_unique_one_side": True,
            "notes": "Power BI relationship: dimension one-side to mart many-side; Single and Active.",
        }
        for spec in RELATIONSHIP_SPECS
    ]


def artifact_contract_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for name in EXPECTED_MARTS:
        rows.append(
            {
                "artifact_type": "mart",
                "artifact_name": name,
                "primary_or_grain_key": ";".join(MART_GRAIN_KEYS[name]),
                "required_columns": ";".join(REQUIRED_ARTIFACT_COLUMNS[name]),
                "output_subdirectory": "marts",
                "output_format": "parquet",
            }
        )
    for name in EXPECTED_DIMENSIONS:
        rows.append(
            {
                "artifact_type": "dimension",
                "artifact_name": name,
                "primary_or_grain_key": DIMENSION_KEYS[name],
                "required_columns": ";".join(REQUIRED_ARTIFACT_COLUMNS[name]),
                "output_subdirectory": "dimensions",
                "output_format": "parquet",
            }
        )
    return rows
