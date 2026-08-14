# Technical Report - Iran Housing Market Intelligence

- **Report version:** `final-technical-report-v3.3-public-fastapi-deployment`
- **Generated at (UTC):** `2026-08-14T01:30:30.947849+00:00`
- **Silver Master rows:** **1,000,000**
- **Sale asking-price-per-sqm eligible rows:** **405,809**
- **Rental-equivalent eligible rows:** **331,739**
- **Gold status:** Gold QA status=`PASS`, critical_failures=`0`, gold_data_contract_ready=`True`, marts=`10`, dimensions=`5`, physical_relationships=`13`.

## 1. Analytical architecture and reproducibility

The project uses a canonical, read-only Silver Master as the analytical source of truth. Cleaning, eligibility, duplicate control, outlier handling, spatial QA, modeling, text analysis, segmentation, and Gold construction are implemented in versioned modules under `src/`. The final notebook is an orchestrator: it runs or validates those modules, loads accepted artifacts, and presents evidence without duplicating business logic inside notebook cells.

Gold is the report/dashboard-ready layer derived from Silver and accepted canonical analytical outputs. The final report builders do not refit models or replace upstream calculations.

### Pipeline readiness metadata

| Stage | Status | Ready | Components | Critical failures | Reviews |
|---|---|---|---|---|---|
| M1 | PASS | True | 1 | 0 | 0 |
| M2 | REVIEW | True | 1 | 0 | 1 |
| M3 | PASS | True | 7 | 0 | 0 |
| M4 | PASS | True | 2 | 0 | 0 |

## 2. Data-quality audit

The current M2 quality gate contains **16** checks: **15 PASS**, **1 REVIEW**, **0 FAIL**, and **0 critical failures**.

| Check | Actual | Expected | Status | Critical |
|---|---|---|---|---|
| required_silver_columns | 0 | 0 | PASS | True |
| source_row_id_unique | 1000000 | 1000000 | PASS | True |
| price_regime_null_rows | 0 | 0 | PASS | True |
| price_unit_null_rows | 0 | 0 | PASS | True |
| price_observation_type_null_rows | 0 | 0 | PASS | True |
| silver_master_version_null_rows | 0 | 0 | PASS | True |
| nonfinite_sale_psm_rows | 0 | 0 | PASS | True |
| nonfinite_rent_equivalent_rows | 0 | 0 | PASS | True |
| exact_duplicate_excess_rows_kept | 0 | 0 | PASS | True |
| high_probable_duplicate_excess_rows_kept | 0 | 0 | PASS | True |
| sale_psm_eligible_rows | 405809 | >0 | PASS | True |
| rent_eligible_rows | 331739 | >0 | PASS | True |
| supply_eligible_rows | 966028 | >0 | PASS | True |
| map_eligible_rows | 637815 | >0 | PASS | False |
| currency_source_confirmation | toman_assumed_unconfirmed | toman_assumed_unconfirmed | REVIEW | False |
| price_observation_contract | asking_price | asking_price | PASS | True |

The audit verifies core contract properties such as source-row uniqueness, non-null price-regime/unit/observation metadata, finite final metrics, duplicate retention policy, and eligible analytical populations. A non-critical REVIEW is a documented limitation rather than an analytical failure.

### 2.1 Persian/Arabic character and structured-field standardization

M2 standardization is lineage-preserving: the original source columns remain available for auditability while derived normalized/typed columns are added for analysis. For listing text, Arabic Yeh/Kaf glyph variants are mapped to their canonical Persian forms; Persian and Arabic-Indic digits are converted to ASCII/Western digits 0-9; non-breaking/unusual spaces are normalized; repeated whitespace is collapsed; and leading/trailing whitespace is removed. NULL values remain NULL. The normalized title and description are derived fields and do not overwrite the raw text.

Structured-field standardization separately normalizes missing-like strings, removes supported numeric separators, parses monetary/area/coordinate/capacity fields into typed columns, maps supported boolean values, preserves censored or source-semantic categories through explicit flags, and records unresolved parsing in `type_parse_error_count` rather than inventing values.

| Standardization metric | Value | Status | Interpretation |
|---|---|---|---|
| row_count | 1,000,000 | PASS | Rows covered by canonical M2 standardization. |
| title_normalized_changed_rows | 516,041 | PASS | Rows where the derived normalized title differs from the preserved raw title because of character, digit, or whitespace normalization. |
| description_normalized_changed_rows | 887,874 | PASS | Rows where the derived normalized description differs from the preserved raw description because of character, digit, or whitespace normalization. |
| rows_with_parse_error | 724 | REVIEW | Rows with at least one structured-field parse issue; retained and flagged for review rather than silently coerced. |
| analysis_month_parse_failure_rows | 0 | PASS | Rows whose listing month could not be parsed; final expectation is zero. |

A `REVIEW` on residual structured parse errors is not equivalent to row deletion or imputation: those source values remain auditable and downstream eligibility rules decide whether a record is usable for a specific metric.

### 2.2 Missingness policy

The frozen project policy retains NULL in Silver, forbids blanket zero/False filling and blanket row deletion, and permits imputation only inside a justified downstream model/analysis workflow. The current action table documents **196** column-level issue/action rows. Task-specific NULL preservation is used for **109** rows; model-only numeric imputation for **37**; low-coverage retention for **25**; and explicit unknown-vs-False amenity preservation for **17**.

Missing values are never globally interpreted as zero or False. Structural non-applicability is retained as NULL. Model-specific imputation is performed only inside the relevant training/resampling workflow to avoid leakage.

### 2.3 Spatial quality

Spatial QA covered **1,000,000** records; **655,608** had coordinate pairs and **655,584** fell inside the accepted Iran validation window. Advanced QA used a point-in-polygon sample of **5,000** with an inside-boundary rate of **99.4%**, plus a reverse-geocoding sample of **80** with Iran-country match **100.0%** and city/county match **58.8%**. The Advanced Spatial Analysis bonus evidence is **READY**: the external boundary point-in-polygon test and reverse-geocode request/country integrity are the bonus gates. City/county string alignment remains a separate diagnostic because the external administrative taxonomy and platform city taxonomy are not identical; its threshold is not lowered to force a pass. Exact coordinates are not published in Gold or the dashboard.

## 3. Duplicate control and Advanced Entity Resolution

The upstream standardization and missingness policies above are lineage-preserving and eligibility-driven. Raw/audit information remains available in Silver while typed, cleaned, derived, quality-flag, price-regime, final-metric, and eligibility fields define analytical use. Records are not physically deleted simply because they fail a downstream metric rule.

Duplicate handling is conservative: same-month duplicate excess is removed from supply eligibility, while cross-month repeats remain retained and auditable because a repeated listing across months may represent continued platform activity rather than a duplicate observation at one time point.

Advanced Entity Resolution evidence is **READY**. The canonical duplicate module uses `deterministic_multi_pass_record_linkage` with exact, high, and medium confidence tiers, stable probable-duplicate cluster IDs, and cross-month linkage that is retained rather than automatically deleted. Candidate rows: **3,857**; linked clusters: **1,916**; multi-month clusters: **1,692**; maximum cluster size: **8**; cluster-ID coverage: **100.0%**. Only exact/high-confidence same-month excess can affect supply eligibility; medium and cross-month links remain audit evidence.

| Duplicate metric | Value |
|---|---|
| row_count | 1,000,000 |
| exact_duplicate_rows | 0 |
| exact_duplicate_excess_rows | 0 |
| high_probable_duplicate_rows | 278 |
| high_probable_duplicate_excess_rows | 144 |
| medium_candidate_rows | 197 |
| cross_month_repeat_rows | 3,399 |
| conservative_supply_rows | 999,856 |
| entity_resolution_candidate_rows | 3,857 |
| entity_resolution_clustered_candidate_rows | 3,857 |
| entity_resolution_cluster_count | 1,916 |
| entity_resolution_method | deterministic_multi_pass_record_linkage |
| entity_resolution_cluster_coverage_rate | 1 |
| entity_resolution_max_cluster_size | 8 |
| entity_resolution_multi_month_cluster_count | 1,692 |
| advanced_entity_resolution_bonus_ready | true |
| entity_resolution_automatic_exclusion_policy | exact_or_high_same_month_excess_only |
| entity_resolution_cross_month_policy | retain_and_flag |

## 4. Outlier rules

Canonical outlier handling is **flag-based, not row-deletion-based**. The frozen M2-05 rule version is `outlier-policy-m2-v2`. Positive relevant values use group-aware IQR thresholds with a canonical multiplier of **3.0 x IQR**. Local thresholds require at least **50** rows; the broader property-category fallback requires at least **200** rows.

For area and sale asking-price-per-sqm metrics, the canonical grouping is city x property category with property-category fallback. Rental and deposit thresholds additionally respect price regime. Groups without sufficient threshold support are retained and reported rather than assigned an arbitrary global threshold. Construction-year pre-1370 values are treated as censored source semantics/review information, not automatically as data errors.

| Metric / flag | Rows | Status |
|---|---|---|
| dataset_rows | 1,000,000 | PASS |
| primary_area_outlier_flag | 25,093 | PASS |
| building_area_outlier_flag | 18,722 | PASS |
| land_area_outlier_flag | 21,550 | PASS |
| sale_price_per_sqm_outlier_flag | 10,024 | PASS |
| monthly_rent_outlier_flag | 10,826 | PASS |
| deposit_outlier_flag | 10,046 | PASS |
| outlier_area_flag | 37,914 | PASS |
| outlier_price_flag | 28,164 | PASS |
| outlier_year_flag | 20,665 | PASS |

Sensitivity evidence compares context-eligible canonical flags with a global **2.0 x IQR** rule and a log-MAD alternative. The canonical counts in this sensitivity table can therefore differ slightly from raw Silver flag counts when a raw flag falls outside the final analytical context. Different methods identify different numbers of unusual observations, so no alternative threshold is treated as ground truth.

| Metric | Evaluated | Canonical | IQR x 2 | Log-MAD |
|---|---|---|---|---|
| sale_price_per_sqm | 428,410 | 9,840 | 26,890 | 23,589 |
| monthly_rent | 290,881 | 10,726 | 26,641 | 1,412 |
| deposit | 347,727 | 9,917 | 24,240 | 7,020 |

## 5. Price-regime separation

The canonical taxonomy is version `price-regime-v1` and contains all **9** regimes, including non-analytical service/unknown states. The complete taxonomy is retained even when a regime has zero observed rows so that classification behavior is explicit and reproducible.

| Regime | Observed rows | Validation | Cleaning / interpretation rule |
|---|---|---|---|
| sale | 597,569 | PASS | Keep fixed positive sale asking price; non-fixed sale values stay non-analytical. |
| rent_plus_deposit | 289,100 | PASS | Preserve both fixed components. |
| full_deposit | 59,222 | PASS | Preserve semantic zero rent and positive deposit. |
| rent_only | 2,859 | PASS | Preserve positive rent and semantic zero deposit. |
| rent_negotiable | 1,801 | REVIEW | Do not invent numeric values; retain for review. |
| rent_unknown_or_incomplete | 143 | REVIEW | Do not force incomplete records into a numeric rent regime. |
| temporary_rent | 29,903 | PASS | Keep separate from long-term rent. |
| service | 19,403 | PASS | Keep outside property price analysis. |
| unknown | 0 | PASS | Retain and review; do not guess a regime. |

`REVIEW` is expected for negotiable or incomplete monetary structures because numeric values are intentionally not invented. `service` and `temporary_rent` remain separate from long-term property-price analysis, and `unknown` is never coerced to another regime.

This separation prevents sale prices from being combined directly with rent/deposit components in a single monetary metric. Every downstream analytical population must declare a compatible price regime.

## 6. Deposit-to-rent conversion

For validated long-term rental records, the project publishes three sensitivity scenarios. If `R` is monthly rent and `D` is deposit:

- Low: `R + D x 0.025`
- Base: `R + D x 0.030`
- High: `R + D x 0.035`

These rates are derived from `config/settings.yaml` and correspond to **25,000 / 30,000 / 35,000 Toman of monthly-rent equivalent per 1,000,000 Toman of deposit**. They are project sensitivity assumptions, not an externally verified official market conversion rate. Semantic zero is preserved for monthly rent in full-deposit listings and for deposit in rent-only listings.

## 7. Sale asking price per square meter

The canonical comparable sale metric is published only for eligible records:

`sale_price_per_sqm = validated sale asking price / primary_area_sqm`

`primary_area_sqm` follows the upstream property-compatible area contract. Outside the valid analytical population, the final metric remains NULL. The observation is an asking price, not a transaction price.

| Final metric | Eligible | Populated | Mismatch | Status |
|---|---|---|---|---|
| sale_price_per_sqm | 405,809 | 405,809 | 0 | PASS |
| rent_equivalent_sensitivity | 331,739 | 331,739 | 0 | PASS |

A zero mismatch count is required for consistency between eligibility flags and populated final metrics.

## 8. Market segmentation method

The final segmentation release is a **descriptive market typology**, not a causal class, socioeconomic class, or buyer-persona model. Published method(s): `compatible_domain_segment, rule_based_descriptive_typology`. Manifest primary_method=`rule_based_descriptive_typology`. The profile contains **405,809** assigned listings.

| Segment | English label | Method | N | Share |
|---|---|---|---|---|
| SEG02 | Mid-market \| family | rule_based_descriptive_typology | 145,110 | 35.8% |
| SEG04 | Land / investment property | compatible_domain_segment | 95,119 | 23.4% |
| SEG01 | Relative affordable \| small urban unit | rule_based_descriptive_typology | 87,065 | 21.5% |
| SEG03 | Relative luxury \| family | rule_based_descriptive_typology | 58,044 | 14.3% |
| SEG05 | Commercial / office property | compatible_domain_segment | 20,471 | 5.0% |

Incompatible property families are not forced into one apartment-style geometry. Relative affordable/luxury labels are meaningful only within the compatible local/reference pricing framework.

## 9. Price Drivers / AVM interpretation

Price-driver outputs use a controlled Ridge model with held-out evaluation. Adjusted effects are interpreted as **model-implied adjusted associations/contrasts**. Permutation importance is interpreted as **held-out predictive contribution**. Neither output is a causal effect. Location and property-category blocks are structural controls and should be separated from potentially actionable property characteristics. On the held-out test split, R2(log)=`0.277`, median APE=`33.6%`, N=`53,615`.

## 10. Main limitations

1. **Asking-price data:** listing asks are not realized transaction prices.
2. **Currency contract:** Toman is the operational unit, but the source currency remains unconfirmed; no undocumented factor-of-ten conversion is applied.
3. **Platform selection bias:** the dataset is not the complete Iranian housing stock, inventory, or transaction universe.
4. **Market Temperature:** the index is a proxy for asking-price and listing-activity trends; it is not liquidity, absorption, physical inventory, or supply tightness.
5. **Model interpretation:** adjusted associations and permutation importance are non-causal and constrained by model error and feature coverage.
6. **Outlier policy:** IQR rules identify unusual observations, not confirmed errors; source rows remain auditable in Silver.
7. **Spatial privacy and quality:** exact coordinates are not exposed in Gold or the dashboard; surfaced spatial results are aggregated.
8. **Segmentation:** the final segments are descriptive market types, not socioeconomic classes, buyer personas, or causal groups.
9. **Standardization scope:** character normalization is orthographic/technical and does not rewrite listing semantics; raw text and unresolved parse evidence remain auditable.
10. **Missingness:** missing values are not blanket-filled with zero/False; feature applicability varies across property families.
11. **Temporal scope:** findings describe the project's core analytical months and should not be treated as a permanent long-run market regime.

## 11. Public FastAPI deployment - bonus evidence

The accepted Gold layer is exposed through a public, read-only FastAPI service deployed on Render. Deployment evidence was validated on **2026-08-14**.

- **Base URL:** `https://ihmi-fastapi.onrender.com`
- **Interactive Swagger/OpenAPI:** `https://ihmi-fastapi.onrender.com/docs`
- **Health endpoint:** `https://ihmi-fastapi.onrender.com/health`
- **Public smoke test:** `PASS` with `health=ok`, `gold_qa_status=PASS`, `marts=10`, and `dimensions=5`
- **Swagger availability:** `/docs` returned HTTP `200`

The API is an access layer over accepted canonical Gold artifacts. It does not rebuild Silver or Gold, refit models, or recompute Market Temperature or segmentation. Exact listing coordinates are not exposed. The same asking-price, listing-activity, observational/non-causal, AVM-prototype, and descriptive-segmentation interpretation boundaries remain in force.

## 12. Final data-product status

Gold QA status=`PASS`, critical_failures=`0`, gold_data_contract_ready=`True`, marts=`10`, dimensions=`5`, physical_relationships=`13`.

This Technical Report is designed to accompany the Executive Summary, the Restart-and-Run-All final notebook, the final dashboard, and the public FastAPI deployment in the delivery package.

### Final reporting surface

The professor-facing reporting layer is intentionally limited to two consolidated Markdown reports: `Executive_Summary.md` and `Technical_Report.md`. Task-level QA/technical notes remain internal evidence and are not additional final reports. Product Recommendations and the demonstration are separate delivery artifacts rather than duplicate technical reports.

---

### Canonical artifacts used

- `outputs/tables/milestone_2/quality_gate/quality_gate_summary.csv`
- `outputs/tables/milestone_2/standardization/standardization_summary.csv`
- `outputs/tables/milestone_2/outliers/outlier_summary.csv`
- `outputs/tables/milestone_2/outliers/outlier_sensitivity.csv`
- `outputs/tables/milestone_2/duplicates/duplicate_summary.csv`
- `outputs/tables/milestone_2/duplicates/duplicate_supply_impact.csv`
- `outputs/tables/milestone_2/final_metrics/final_metric_summary.csv`
- `outputs/tables/milestone_2/price_regimes/price_regime_review_summary.csv`
- `outputs/tables/milestone_3/market_segmentation/segment_profile.csv`
- `outputs/tables/milestone_2/missingness/missingness_action_table.csv`
- `config/settings.yaml`
- `outputs/tables/milestone_3/spatial_quality/spatial_quality_summary.csv`
- `outputs/qa/milestone_3/market_segmentation/segmentation_manifest.json`
- `outputs/tables/milestone_3/price_drivers/price_driver_model_diagnostics.csv`
- `data/gold/metadata/dashboard_quality_status.csv`
- `data/gold/qa/gold_qa_manifest.json`

Outlier configuration source: `config/settings.yaml`.
Rent/deposit sensitivity configuration source: `config/settings.yaml`.

This report builder does not refit statistical or machine-learning models; it summarizes accepted project artifacts and frozen methodological contracts.
