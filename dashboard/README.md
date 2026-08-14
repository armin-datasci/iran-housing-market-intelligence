# Dashboard Workspace

This directory contains the **Power BI-specific semantic/design layer**. It is downstream of the frozen Gold data contract.

## Canonical Power BI data sources

Power BI must read Parquet directly from:

```text
data/gold/marts/
data/gold/dimensions/
```

Do not use `data/gold/reporting/*.csv` in the final dashboard. Those files are inspection duplicates and may be removed after Parquet source verification.

## Source-of-truth boundary

Gold structural metadata remains under `data/gold/metadata/` because Gold QA uses it to validate the semantic handoff. In particular, these files are **not moved** out of Gold:

```text
data/gold/metadata/gold_artifact_contract.csv
data/gold/metadata/gold_relationship_contract.csv
data/gold/metadata/dashboard_page_registry.csv
data/gold/metadata/dashboard_quality_status.csv
data/gold/metadata/gold_source_lineage.csv
data/gold/metadata/gold_artifact_inventory.csv
```

Dashboard-specific Measure/DAX/filter/page artifacts belong under `dashboard/contracts/`. This keeps a single canonical location for each concern and avoids duplicating Gold QA metadata.

## Frozen model contract

- 10 Gold marts
- 5 dimensions
- 13 active single-direction physical relationships
- `dim_user_type` is semantic-only/disconnected
- no exact coordinates in the semantic model
- Power BI uses Parquet

## Official pages

1. P01 Executive Market Overview
2. P02 Data Quality
3. P03 Price Map
4. P04 Supply and Price Trends
5. P05 Amenities and Price Drivers
6. P06 Seller Type Comparison
7. P07 Text Signals
8. P08 Market Segments

The later dashboard handoff bundle should combine the Parquet schema/inventory, Gold structural contracts, the files under `dashboard/contracts/`, and a short design-team README without copying upstream M1-M3 engineering QA.
