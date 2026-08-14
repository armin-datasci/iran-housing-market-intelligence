# IHMI Dashboard Team Handoff

## Purpose

This bundle is the implementation contract for the final Power BI dashboard. It is downstream of the frozen Gold analytical layer and must not recreate M1-M3 cleaning/modeling logic.

## Required source files

Power BI reads:

```text
data/gold/marts/*.parquet       # exactly 10 marts
data/gold/dimensions/*.parquet  # exactly 5 dimensions
data/gold/metadata/dashboard_quality_status.csv  # P02 metadata
```

Gold structural evidence travels with the handoff but is not a report fact table:

```text
data/gold/metadata/gold_artifact_contract.csv
data/gold/metadata/gold_relationship_contract.csv
data/gold/metadata/dashboard_page_registry.csv
data/gold/metadata/gold_source_lineage.csv
data/gold/metadata/gold_artifact_inventory.csv
data/gold/qa/gold_qa_manifest.json
data/gold/qa/gold_qa_checks.csv
```

## Build order

1. Create Power Query text parameter `pProjectRoot`.
2. Load the 10 mart Parquets, 5 dimension Parquets, and `dashboard_quality_status.csv` using `power_query_templates.pq`.
3. Create `z_Measures` and `p_MinValidN` from `calculated_tables.dax`.
4. Add `dim_location[Geocode Location]` from `calculated_columns.dax`; set Data Category = Place.
5. Build exactly the 13 relationships in `dashboard_relationship_contract.csv`.
6. Keep `dim_user_type` disconnected; do not create `dim_segment`.
7. Create measures from `measures.dax` and assign formats/display folders from `dashboard_measure_catalog.csv`.
8. Build exactly P01-P08 using `dashboard_page_contract.csv` and `dashboard_visual_contract.csv`.
9. Apply filter behavior from `dashboard_filter_contract.csv`; a mandatory filter may be fixed or non-applicable on a page if the upstream estimate does not support it.
10. Reconcile the completed PBIX against `dashboard_reference_metrics.csv` and the QA checklist before release.

## Official pages

- P01 Executive Market Overview
- P02 Data Quality
- P03 Price Map
- P04 Supply and Price Trends
- P05 Amenities and Price Drivers
- P06 Seller Type Comparison
- P07 Text Signals
- P08 Market Segments

Do not add a ninth analytical page. Hot/Cold belongs in P03 with a compact summary on P01; AVM quality/error evidence belongs in P05; advanced reliability belongs in P02/tooltip context.

## Required filters

City, Neighborhood, Property, Month, Price Regime, User Type, Minimum Valid Listings. Applicability is governed by `dashboard_filter_contract.csv`; do not manufacture unsupported slicer behavior.

## Final release gates

- Gold QA = PASS and `gold_data_contract_ready=true`.
- Exactly 10 mart Parquets + 5 dimension Parquets.
- Exactly 13 active single-direction physical relationships.
- No exact coordinate fields.
- `dim_user_type` disconnected.
- Reference metrics reconcile within tolerance.
- All eight pages show appropriate reliability/sample context.
- Asking-price and Toman-assumption caveats visible on financial pages.
