# Technical Report - Iran Housing Market Intelligence

- **Report version:** `final-technical-report-v4.0-stage2-complete`
- **Analytical snapshot:** accepted pipeline/Gold evidence through `2026-08-14`
- **Documentation revision:** `2026-08-20`
- **Silver Master rows:** **1,000,000**
- **Sale asking-price-per-sqm eligible rows:** **405,809**
- **Rental-equivalent eligible rows:** **331,739**
- **Supply-analysis eligible rows:** **966,028**
- **Map-analysis eligible rows:** **637,815**
- **Gold status:** `PASS`; critical failures=`0`; data contract ready=`True`; architecture=`10 marts / 5 dimensions / 13 physical relationships`

## 1. Analytical architecture, technology stack, and reproducibility

### 1.1 Objective and design

IHMI is implemented as a reproducible analytical data product rather than as a notebook-only analysis. The canonical architecture contains two persisted analytical layers:

1. **Silver Master** - the read-only row-level analytical source of truth, preserving source/audit fields together with typed, cleaned, derived, quality, eligibility, version and lineage fields.
2. **Gold** - compact dashboard/report/API-ready marts and dimensions derived only from Silver and accepted canonical analytical outputs.

Raw input remains an external/source input and is not treated as a third persisted analytical layer. The final notebook is an **evidence/presentation and orchestration surface**; business logic lives in versioned modules under `src/` and is executed by the canonical pipeline runner. The notebook does not maintain a second implementation of cleaning, feature engineering, modeling or Gold logic.

### 1.2 Technology stack

- **Core language / analytics:** Python, Polars, pandas, NumPy, PyArrow
- **Statistics / machine learning:** SciPy, scikit-learn, joblib
- **Visualization:** Matplotlib
- **Configuration / reproducibility QA:** PyYAML, psutil
- **Notebook / validation:** Jupyter, pytest
- **BI / semantic layer:** Power BI
- **Application / API:** Streamlit dashboard, FastAPI, Uvicorn

Direct dependencies are declared in `requirements.txt`. Exact resolved runtime versions for the accepted environment are recorded in `outputs/qa/final_pipeline/environment_versions.json` rather than being guessed or retrospectively pinned.

### 1.3 Execution and artifact contract

The canonical runner is `scripts/run_pipeline.py`. Final reporting reads accepted upstream artifacts and **does not refit models**. Heavy runtime data are intentionally not normal Git-tracked assets. The documented restore model is:

`external_data/real_estate_ads.csv` -> `data/silver/silver_master.parquet` -> accepted analytical outputs -> `data/gold/`

The reproducibility and delivery inventory is documented in `docs/DATA_AND_ARTIFACTS.md`, including Raw, Silver, Gold, CSV/Parquet outputs, model artifacts, QA evidence, spatial references and the frozen runtime bundle.

### 1.4 Pipeline readiness

| Stage | Status | Ready | Critical failures | Interpretation |
|---|---|---:|---:|---|
| M1 | PASS | True | 0 | Source loading/documentation/audit accepted. |
| M2 | REVIEW | True | 0 | One documented non-critical currency-source review remains. |
| M3 | PASS | True | 0 | Canonical analyses accepted. |
| M4 | PASS | True | 0 | Gold and semantic handoff accepted. |

**Limitation.** Reproducibility of the full million-row workflow depends on access to the frozen runtime/data bundle because heavy Raw/Silver/Parquet artifacts are not intended to be committed to ordinary Git history.

## 2. Data-quality and canonical cleaning layer

### 2.1 Standardization and structured parsing

**Question.** Can heterogeneous Persian/Arabic text and structured listing fields be normalized without overwriting source evidence or inventing values?

**Input / population.** All **1,000,000** Silver rows and the full source schema.

**Method.** Original fields are preserved. Derived text normalization maps Arabic/Persian Yeh/Kaf variants, converts Persian/Arabic-Indic digits to ASCII, normalizes unusual spaces, collapses repeated whitespace and trims boundaries. Structured parsing separately handles supported numeric separators, monetary/area/coordinate/capacity fields and booleans. Missing-like strings are treated by column-specific rules rather than a global sentinel rule.

**Evidence.** `title_normalized_changed_rows=516,041`; `description_normalized_changed_rows=887,874`; `rows_with_parse_error=724` (`REVIEW`); `analysis_month_parse_failure_rows=0` (`PASS`).

**Decision.** Preserve raw text and source values for auditability; publish normalized/typed fields as derived analytical columns; retain unresolved parsing as explicit review evidence rather than silent coercion.

**Limitation.** Orthographic/technical normalization does not rewrite listing semantics, resolve every colloquial expression, or infer missing values.

### 2.2 Missingness policy

**Question.** How should missing and structurally non-applicable values be handled without introducing false zeros or leakage?

**Input / population.** The complete Silver schema and downstream model-specific feature sets.

**Method.** NULL is preserved in Silver. Blanket `0`, `False`, row deletion, or global imputation is prohibited. Model-only numeric imputation is performed inside the relevant training/resampling workflow; amenity unknown/non-applicable states are not automatically collapsed to `False`.

**Evidence.** The frozen action table documents **196** column-level issue/action rows, including task-specific NULL preservation, model-only imputation, low-coverage retention and explicit unknown-versus-False handling.

**Decision.** Missingness is a data contract and feature-applicability issue, not a cosmetic cleaning problem.

**Limitation.** Some property-family fields have structurally different applicability and coverage; downstream model estimates remain conditional on observed feature availability.

### 2.3 Duplicate control and Advanced Entity Resolution

**Question.** Which repeated records can safely reduce listing-activity counts, and which should remain auditable rather than be deleted?

**Input / population.** **1,000,000** records; deterministic exact/probable duplicate evidence and cross-month repeat candidates.

**Method.** The canonical duplicate task uses deterministic multi-pass record linkage with exact, high and medium confidence tiers plus stable probable-duplicate cluster IDs. Only exact/high-confidence **same-month excess** can reduce supply eligibility. Medium-confidence candidates and cross-month links remain retained and flagged because repeated listings across months can represent continued platform activity.

**Evidence.** Exact duplicate rows/excess=`0/0`; high-probable rows/excess=`278/144`; medium candidates=`197`; cross-month repeats=`3,399`; conservative supply=`999,856`. Advanced ER includes **3,857 candidate rows**, **1,916 linked clusters**, **1,692 multi-month clusters**, maximum cluster size **8**, and **100% cluster-ID coverage**. Advanced Entity Resolution bonus evidence is ready.

**Decision.** Use conservative exclusion for same-month listing-activity measurement while preserving uncertain and longitudinal repeat evidence for auditability.

**Limitation.** Record linkage is evidence about probable listing identity, not proof that two platform records represent the same physical market event or transaction.

### 2.4 Group-aware outlier policy

**Question.** How can extreme values be controlled without treating every unusual listing as an error?

**Input / population.** Valid positive area, sale asking PSM, monthly rent and deposit observations, with property/regime-aware grouping.

**Method.** Frozen version `outlier-policy-m2-v2` uses group-aware **3 x IQR** as the canonical flag rule, local minimum **N=50**, property-category fallback **N=200**, and flag-only treatment. **2 x IQR** and log-MAD are retained as sensitivity methods.

**Evidence.** Canonical flags include `sale_price_per_sqm_outlier_flag=10,024`, `monthly_rent_outlier_flag=10,826`, `deposit_outlier_flag=10,046`, `outlier_area_flag=37,914`, `outlier_price_flag=28,164`, and `outlier_year_flag=20,665`.

**Decision.** No source row is deleted merely because it is statistically unusual; downstream eligibility and task-specific quality rules determine analytical use.

**Limitation.** IQR/MAD rules identify unusual observations, not confirmed data errors. Thin local groups may require fallback thresholds or remain unsupported.

### 2.5 Spatial quality and advanced external validation

**Question.** Are coordinates sufficiently coherent for aggregate spatial analysis, and can that conclusion be checked against an external geographic reference without publishing exact listing locations?

**Input / population.** All **1,000,000** rows; **655,608** coordinate pairs.

**Method.** The canonical spatial workflow performs world/Iran-range validation, partial-coordinate checks, likely-swapped-coordinate checks, aggregation eligibility, external ADM2 point-in-polygon validation and a small stratified reverse-geocode validation. The external boundary is the Iran **ADM2/Shahrestan** geoBoundaries reference (`IRN-ADM2-26516999`, boundary year represented 2017; source OpenStreetMap/Wambacher; ODbL). Boundary validation uses a deterministic city-stratified sample of unique coordinate pairs rather than bulk geocoding. Reverse geocoding uses a small deterministic city-stratified sample with extra coverage for Tehran, Mashhad, Karaj and Isfahan.

**Evidence.** Coordinate pairs=`655,608`; Iran-window-valid=`655,584`; likely-swapped rows=`0`; geo-aggregate eligible=`655,114` in the spatial QA artifact. This spatial-QA eligibility is broader than the final **map-analysis population of 637,815**, which also applies downstream analytical population and quality criteria. The advanced point-in-polygon test sampled **5,000** coordinates: **4,970** were inside the reference boundary, an inside rate of **99.4%**, above the predefined 98% gate. The reverse-geocode sample contained **80** points; request success and Iran-country match were both **100%**. City/county string match was **58.8% (47/80)** and remains `REVIEW`, not `FAIL`.

**Decision.** Spatial integrity is sufficient for aggregate mapping and the Advanced Spatial Analysis bonus is ready. The lower locality-string match is treated as a taxonomy-alignment diagnostic because the platform's city naming and an external administrative hierarchy are not one-to-one; the threshold is not lowered post hoc to manufacture a pass.

**Limitation.** External boundaries and administrative names have their own vintage/taxonomy limitations. Exact coordinates are privacy-sensitive and are therefore excluded from Gold, dashboards and public APIs; published spatial evidence is aggregated.

### 2.6 Monetary-scale and currency decision

**Question.** Should raw monetary values be globally multiplied/divided by 10, and what can the data actually establish about the absolute Rial/Toman denomination?

**Input / population.** The currency audit uses all **1,000,000** rows; **985,725** are in the May-December 2024 core period. Eleven available raw/typed monetary field pairs and independent sale/rent/deposit populations are evaluated.

**Method.** The decision combines two evidence layers. **Scale integrity** checks raw-to-typed parity and searches for local `0.1x/10x` anomaly subpopulations in comparable neighborhood/property cells. **Economic coherence** independently evaluates the source rent-credit transformation, area-normalized sale/rent/deposit scales and cross-market sale-versus-rent-equivalent relationships.

**Evidence.** Raw-to-typed monetary mismatches=`0` across **11** pairs. Near-`0.1x / 10x` shares are **0.126% / 0.049%** for apartment-sale PSM, **1.119% / 0.044%** for apartment rent per sqm and **0.392% / 0.020%** for apartment deposit per sqm; there is no material broad factor-of-ten mixture. In **25,400** source rent-credit transformations, the median absolute delta ratio is **0.030**; **55.244%** are exactly `0.030` and **86.343%** fall in `0.025-0.035`. Across **603** neighborhoods with at least 30 sale and 30 rental observations, the log-correlation between apartment sale asking PSM and the `0.030` monthly rent-equivalent PSM is **0.9630**. The median sale-to-monthly-equivalent ratio is **236.05 months** (P05 **172.95**, P95 **314.54**).

**Decision.** The supplied **numerical monetary scale is internally coherent and remains unchanged (`scale=1`)**. No project-wide `x10` or `/10` conversion is justified. Toman is used as the operational reporting unit, while the formal source-status label remains **`toman_assumed_unconfirmed`**.

**Limitation.** Internal economic relationships can reject a broad mixed-scale error but cannot identify the absolute denomination under a global rescaling: if every monetary field is multiplied by the same factor, ratios, correlations and rankings can remain largely unchanged. An authoritative source-unit document would therefore supersede the operational label through a versioned migration and parity review. All monetary observations remain listing asking prices, not transaction prices.

### 2.7 Price-regime separation and rental equivalence

**Question.** Can sale prices, long-term rent/deposit structures, temporary rent and non-property service records be represented without mixing incompatible monetary concepts?

**Input / population.** All source rows classified into the frozen nine-state price-regime taxonomy.

**Method.** Sale, `rent_plus_deposit`, `full_deposit`, `rent_only`, negotiable/incomplete rent, temporary rent, service and unknown states remain explicit. Numeric values are not invented for negotiable/incomplete cases. For compatible long-term rentals, the project publishes three sensitivity scenarios: `R + D*0.025`, `R + D*0.030`, and `R + D*0.035`.

**Evidence.** Observed counts: sale=`597,569`; rent+deposit=`289,100`; full-deposit=`59,222`; rent-only=`2,859`; negotiable=`1,801`; unknown/incomplete=`143`; temporary rent=`29,903`; service=`19,403`; unknown=`0`. Final sale PSM eligible/populated=`405,809/405,809`, mismatch=`0`; rental-equivalent eligible/populated=`331,739/331,739`, mismatch=`0`.

**Decision.** Every downstream population must declare a compatible regime. The three deposit-to-rent rates are sensitivity assumptions, not official conversion facts.

**Limitation.** Long-term rent/deposit equivalence is a modeling convention and can vary by market practice, time and location; sensitivity scenarios are therefore more defensible than one asserted universal rate.

## 3. Market analysis

### 3.1 National monthly apartment-sale trends

**Definition and objective.** Measure how deduplicated apartment-sale listing activity and median asking price per square meter evolved over the eight-month core period.

**Current status and trend.** Deduplicated national listing activity increased from **34,949** in May to **41,328** in December (**+18.3%**). Median asking PSM moved from **36.36M** to **36.54M Toman/m2** (**+0.5%**). The series is not monotonic: activity softened in August-September before rising strongly in October-December, while the median asking PSM remained comparatively stable.

| Month | Deduplicated apartment-sale listings | Median asking PSM (Toman) |
|---|---:|---:|
| 2024-05 | 34,949 | 36,363,636 |
| 2024-06 | 35,035 | 35,294,118 |
| 2024-07 | 36,560 | 35,300,000 |
| 2024-08 | 36,112 | 35,000,000 |
| 2024-09 | 33,978 | 35,000,000 |
| 2024-10 | 39,594 | 35,321,101 |
| 2024-11 | 41,087 | 35,714,286 |
| 2024-12 | 41,328 | 36,538,462 |

**Geographic / comparative evidence.** Upstream monthly analysis runs across the eligible national/city/neighborhood universe; the four-city restriction is a presentation requirement only where explicitly needed.

**Drivers and contributing factors.** The analysis is descriptive. Changes can reflect platform participation, relisting behavior, seasonality, composition and seller asking behavior in addition to underlying housing-market conditions.

**Market implication / interpretation.** Over this window, platform listing activity expanded much faster than the national median asking PSM. This is evidence of greater listing activity, not proof of increased physical housing supply or completed transaction volume.

**Data limitations and analytical gaps.** Eight months is a short horizon; the platform sample is not the full housing stock; listing asks are not transaction prices; listing counts are not liquidity or absorption.

### 3.2 Listing-Market Temperature Proxy

**Definition and objective.** Rank eligible city/neighborhood markets on a relative proxy combining asking-price trend and listing-activity trend. The primary score uses **60% price trend / 40% listing-activity trend**, with reliability/evidence weighting and explicit direction gates.

**Current status and trend.** The canonical ranking universe is **all eligible entities within each entity level**. HOT requires a positive asking-price trend; COLD requires a negative asking-price trend. Listing-activity sign is informative but not a hard direction gate. The professor-facing neighborhood view requires at least **5 reliable price months**, **5 listing-activity months**, and **N >= 100**.

**Geographic / comparative evidence.** In the required four-city reliable presentation pool, labels are **17 HOT / 30 COLD / 98 NEUTRAL**. Leading HOT examples include Karaj/Golshahrvila (**75.9**), Mashhad/Daneshjoo (**70.0**) and Isfahan/Eshragh (**66.3**). Leading COLD examples include Tehran/Parastar (**-89.0**), Karaj/Andishehnewcity (**-88.4**) and Tehran/North Sohrevardi (**-85.2**).

**Drivers and contributing factors.** The score combines direction and magnitude of asking-price and listing-activity trends, month coverage and trend-direction stability. A price-heavier **70/30** scenario is retained as sensitivity evidence; the predeclared top-5 overlap gate is not relaxed post hoc.

**Market implication / interpretation.** Market Temperature is useful for relative prioritization and exploration of listing-market momentum. It is explicitly **not** a liquidity, absorption, inventory, physical supply tightness or investment-return metric.

**Data limitations and analytical gaps.** Rankings can be sensitive to sample size, time window, platform composition and score weights. HOT/COLD labels are relative states within the eligible ranking universe, not permanent structural classifications.

## 4. Modeling and explanatory analyses

### 4.1 Price Drivers and interpretable AVM / Error Analysis

**Definition and objective.** Estimate adjusted observational contrasts for property characteristics and quantify held-out predictive contribution while explicitly measuring how far the model can be trusted as an AVM-style prototype.

**Current status and trend.** The primary Ridge workflow uses **405,809** sale-PSM-eligible records with time-aware train/validation/test counts of **297,864 / 54,330 / 53,615**. On the held-out test set: `R2(log)=0.277`, `RMSE(log)=1.7093`, Median APE=`33.6%`, P75 APE=`~55.1%`, P90 APE=`~87.8%`, within +/-20%=`29.2%`, within +/-30%=`44.7%`, within +/-50%=`70.4%`, and median absolute PSM error=`~9.15M Toman/m2`.

**Geographic / comparative evidence.** Location is handled as a grouped **City + Neighborhood** block to avoid impossible independent permutations; property family + category are also grouped. Location dominates held-out predictive contribution (`0.2701` increase in RMSE(log)), followed by Rooms (`0.0477`), Building age (`0.0389`) and Area (`0.0339`).

**Drivers and contributing factors.** The controlled Ridge design separates structural controls from potentially interpretable property-characteristic contrasts. Benchmark models (Ridge, Decision Tree and Random Forest) are evaluated on comparable rows/matrices; observed held-out winners are descriptive and are not used to retune on the test set. AVM error analysis is separately surfaced by market subgroups.

**Market implication / interpretation.** Location and structural characteristics contain substantial predictive information, but the model's moderate held-out accuracy sets a clear ceiling on valuation claims. Adjusted effects are **model-implied adjusted associations/contrasts**, not shares of price and not causal effects. Permutation importance is **held-out predictive contribution**, not causal importance.

**Data limitations and analytical gaps.** Model performance is constrained by asking-price measurement, feature coverage, residual confounding and platform selection. The AVM is an interpretable research prototype / initial risk-analysis tool, not a production valuation guarantee or a substitute for transaction comparables and professional appraisal.

### 4.2 Seller-Type Comparison

**Definition and objective.** Evaluate whether agency/business and personal listings show different sale asking PSM after accounting for observed composition.

**Current status and trend.** Accepted population=`125,893`: agency=`114,843`, personal=`11,050`. Raw median agency-versus-personal gap=`+92.0%`. Cross-fitted adjusted difference=`+28.46%`, with 95% interval approximately **+25.97% to +31.00%**. Similar-unit/coarsened-strata sensitivity=`+17.66%`.

**Geographic / comparative evidence.** Controls and robustness strata account for month, city/neighborhood, category/type, area, age, rooms and major amenities where available. Segment-level evidence is exploratory and is not used to redefine the primary estimand.

**Drivers and contributing factors.** The very large raw-to-adjusted shrinkage indicates substantial compositional differences between agency and personal listings. Remaining differences can reflect unobserved property quality, seller strategy, selection into seller type and platform behavior.

**Market implication / interpretation.** Raw seller-group medians should not be read as a direct seller-type premium. Adjusted estimates remain useful as observational comparison evidence after specified controls.

**Data limitations and analytical gaps.** Seller type is not randomly assigned. The adjusted interval is conditional on the fitted control model and does not include full model-refit uncertainty. The result is **not a causal agency premium**.

### 4.3 Precision-validated text signals

**Definition and objective.** Test whether specific listing-language families are associated with asking PSM after controls, while reducing false positives from ambiguous keyword matching.

**Current status and trend.** Six families are carried into the accepted analysis: `new_build`, `unused`, `urgent`, `exchange`, `below_market`, and `migration_sale`. The migration family covers normalized variants such as sale due to migration / because of migration rather than a single exact phrase.

**Geographic / comparative evidence.** The model uses the accepted sale population and control-only residual design rather than raw keyword-price comparisons alone; frequency can vary by month/location, but the primary effect interpretation is controlled and observational.

**Drivers and contributing factors.** Each keyword family is manually precision-validated before primary inference. The accepted analysis uses cross-fitted/held-out control-model residuals, Welch inference and Benjamini-Hochberg FDR control.

| Signal | Manual precision | Adjusted asking-price association | BH-FDR q | Accepted interpretation |
|---|---:|---:|---:|---|
| `new_build` | 89.7% | +13.1% | 0.000137 | Higher after controls |
| `unused` | 100.0% | +9.4% | 0.028006 | Higher after controls |
| `urgent` | 95.7% | -6.5% | 0.056922 | Not FDR-significant |
| `exchange` | 92.3% | -11.0% | 0.000234 | Lower after controls |
| `below_market` | 100.0% | -4.3% | 0.138758 | No clear adjusted evidence |
| `migration_sale` | 100.0% | +0.6% | 0.953955 | No clear adjusted evidence |

**Market implication / interpretation.** Some marketing/property-state language contains controlled price information, but intuitive labels such as "urgent" or "below market" do not automatically imply statistically robust lower asking prices after controls.

**Data limitations and analytical gaps.** Manual precision validates extraction relevance, not causal validity. Promotional language, semantic ambiguity, residual confounding and multiple testing remain limitations. Text features are not used to define the primary market segmentation geometry.

### 4.4 Market Segmentation / Descriptive Typology

**Definition and objective.** Produce interpretable market groups without mixing economically incompatible regimes/property families and without claiming unstable clusters as natural market classes.

**Current status and trend.** The released profile assigns **405,809** sale listings to five descriptive groups:

| Segment | Label | Method | N | Share |
|---|---|---|---:|---:|
| SEG01 | Relative affordable / small urban unit | rule_based_descriptive_typology | 87,065 | 21.5% |
| SEG02 | Mid-market / family | rule_based_descriptive_typology | 145,110 | 35.8% |
| SEG03 | Relative luxury / family | rule_based_descriptive_typology | 58,044 | 14.3% |
| SEG04 | Land / investment property | compatible_domain_segment | 95,119 | 23.4% |
| SEG05 | Commercial / office property | compatible_domain_segment | 20,471 | 5.0% |

**Geographic / comparative evidence.** City is used for compatible local price normalization/profile, not as a raw one-hot driver that would simply recreate geography. Neighborhood is used for profile and local reference, not as the primary clustering geometry.

**Drivers and contributing factors.** For homogeneous apartment-sale candidates, unsupervised clustering is evaluated with scaling/preprocessing, Silhouette, Calinski-Harabasz, Davies-Bouldin, minimum cluster share and resampling ARI. Stability gates are predeclared and are not lowered to force a clustering release. Incompatible property families such as land/commercial are handled through compatible domain segments rather than being forced into apartment geometry.

**Market implication / interpretation.** The final release is intentionally called **Market Types / Descriptive Typology**. It is useful for profiling and BI exploration while remaining honest about the absence of sufficiently stable evidence for stronger latent-cluster claims.

**Data limitations and analytical gaps.** Segments are not socioeconomic classes, buyer personas or causal groups. Relative affordable/luxury labels are meaningful only within the compatible local/reference pricing framework. Non-applicable fields such as Rooms/Age/Amenities must remain N/A for incompatible families rather than being displayed as fake numeric values.

## 5. Statistical governance and claim control

The project consolidates analytical-method, readiness, robustness and design evidence in `outputs/tables/final_reporting/` and `outputs/qa/final_reporting/`. The governance layer documents analytical populations, unit of analysis, main biases, missing-value policy, outlier policy, uncertainty, sensitivity checks, leakage controls and interpretation boundaries.

Key governance rules are:

- asking price != transaction price;
- listing activity != physical inventory, liquidity or absorption;
- Market Temperature is a listing-market proxy;
- adjusted price-driver, seller and text effects are observational associations;
- grouped permutation importance is predictive contribution;
- test-set benchmark winners are descriptive and are not used for post-hoc retuning;
- clustering thresholds are not lowered to force a release;
- fixed model estimates must not appear to refit under unsupported dashboard slicers.

The final notebook surfaces accepted governance evidence but does not fit the final models inside the presentation layer.

## 6. Gold data product, dashboard semantics, and deployment

### 6.1 Gold contract

Gold contains exactly **10 marts**, **5 dimensions** and **13 active single-direction physical relationships**. Marts are:

`mart_market_monthly`, `mart_location_market`, `mart_price_driver_effects`, `mart_price_driver_importance`, `mart_model_quality`, `mart_seller_type`, `mart_text_signals`, `mart_text_monthly`, `mart_segment_profile`, `mart_segment_monthly_mix`.

Dimensions are `dim_location`, `dim_month`, `dim_property`, `dim_price_regime`, and `dim_user_type`. `dim_user_type` is intentionally semantic/disconnected where necessary because the primary seller estimate is a fixed contrast rather than a slicer-refitted model. Exact listing coordinates are excluded.

### 6.2 Dashboard interpretation contract

The Power BI dashboard consumes canonical Gold Parquet artifacts. Visuals are designed to complement Python technical/QA graphics rather than duplicate them. Fixed upstream estimates such as driver effects, seller effects, text effects and model test metrics must preserve their analytical scope and must not be silently recalculated by unsupported slicer combinations.

### 6.3 Public access layers

The accepted Gold layer is exposed through a public, read-only FastAPI deployment on Render:

- Base API: `https://ihmi-fastapi.onrender.com`
- Swagger/OpenAPI: `https://ihmi-fastapi.onrender.com/docs`

The broader project also includes a public Streamlit dashboard. These access layers consume accepted analytical products; they do not rebuild Silver/Gold or refit models. Exact listing coordinates are not exposed.

## 7. Consolidated limitations

1. **Asking-price measurement:** observed monetary outcomes are listing asks, not realized transaction prices.
2. **Currency denomination:** `scale=1` is strongly supported internally, but the source's absolute Rial/Toman denomination remains unconfirmed.
3. **Platform selection:** the dataset is not the complete Iranian housing stock, listing universe or transaction universe.
4. **Short temporal window:** May-December 2024 does not establish a permanent long-run market regime.
5. **Listing activity semantics:** counts measure platform activity after conservative duplicate control; they are not physical inventory, sales volume, liquidity or absorption.
6. **Model error / causality:** AVM and association models have material held-out error and remain observational/predictive rather than causal.
7. **Seller/text confounding:** non-random seller type, promotional language and unobserved listing/property quality can confound adjusted differences.
8. **Spatial taxonomy/vintage:** external ADM2 and platform locality names do not map perfectly; exact coordinates remain private and only aggregate spatial outputs are surfaced.
9. **Outlier uncertainty:** statistical extremeness is not proof of data error; flag-only treatment preserves auditability.
10. **Missingness / applicability:** NULL and structural non-applicability vary by property family and feature; blanket imputation is prohibited.
11. **Segmentation semantics:** released groups are descriptive market types, not guaranteed natural clusters or buyer/socioeconomic classes.
12. **Sensitivity assumptions:** rent/deposit equivalence and Market Temperature weights are declared modeling choices and are accompanied by sensitivity evidence.

## 8. Final data-product status

The accepted delivery package is methodologically aligned across Silver, analytical outputs, Gold, dashboard and API. M1/M3/M4 are `PASS`; M2 remains `REVIEW` while analytically ready because authoritative source denomination is not independently confirmed. This non-critical review is intentionally preserved rather than being converted into a false PASS.

The professor-facing reporting surface remains limited to two consolidated reports: `Executive_Summary.md` and `Technical_Report.md`. Task-level reports, QA checks, manifests, model diagnostics, stability tables and validation samples remain technical evidence/internal appendices rather than competing final reports.

---

### Canonical artifacts used or referenced

**Architecture / reproducibility**
- `README.md`
- `requirements.txt`
- `docs/DATA_AND_ARTIFACTS.md`
- `scripts/run_pipeline.py`
- `outputs/qa/final_pipeline/environment_versions.json`
- `outputs/qa/final_pipeline/final_pipeline_manifest.json`

**Silver / M2 quality**
- `data/silver/silver_master.parquet`
- `outputs/tables/milestone_2/quality_gate/quality_gate_summary.csv`
- `outputs/tables/milestone_2/standardization/standardization_summary.csv`
- `outputs/tables/milestone_2/missingness/missingness_action_table.csv`
- `outputs/tables/milestone_2/duplicates/duplicate_summary.csv`
- `outputs/tables/milestone_2/duplicates/duplicate_supply_impact.csv`
- `outputs/tables/milestone_2/outliers/outlier_summary.csv`
- `outputs/tables/milestone_2/outliers/outlier_sensitivity.csv`
- `outputs/tables/milestone_2/currency/currency_validation_summary.csv`
- `outputs/tables/milestone_2/currency/currency_inference_interpretation.md`
- `outputs/tables/milestone_2/price_regimes/price_regime_review_summary.csv`
- `outputs/tables/milestone_2/final_metrics/final_metric_summary.csv`
- `config/settings.yaml`

**Spatial / M3 market analysis**
- `outputs/tables/milestone_3/spatial_quality/spatial_quality_summary.csv`
- `outputs/qa/milestone_3/spatial_quality/spatial_quality_manifest.json`
- `outputs/qa/milestone_3/spatial_quality/reverse_geocode_validation_sample.csv`
- `external_data/reference/geoboundaries_irn_adm2.geojson`
- `external_data/reference/geoboundaries_irn_adm2_metadata.json`
- `outputs/tables/milestone_3/monthly_market/monthly_market_summary.csv`
- `outputs/tables/milestone_3/market_temperature/market_temperature_summary.csv`
- `outputs/qa/milestone_3/market_temperature/market_temperature_sensitivity.csv`
- `outputs/qa/milestone_3/market_temperature/market_temperature_manifest.json`

**Models / analytical outputs**
- `outputs/tables/milestone_3/price_drivers/price_driver_summary.csv`
- `outputs/tables/milestone_3/price_drivers/price_driver_model_diagnostics.csv`
- `outputs/tables/milestone_3/price_drivers/price_model_benchmark.csv`
- `outputs/tables/milestone_3/price_drivers/price_driver_permutation_importance.csv`
- `outputs/tables/milestone_3/price_drivers/avm_error_analysis.csv`
- `outputs/tables/milestone_3/seller_type_comparison/seller_type_comparison_summary.csv`
- `outputs/tables/milestone_3/seller_type_comparison/seller_type_stratified_summary.csv`
- `outputs/tables/milestone_3/text_price_signals/text_signal_summary.csv`
- `outputs/qa/milestone_3/text_analysis/keyword_manual_validation.csv`
- `outputs/tables/milestone_3/market_segmentation/segment_profile.csv`
- `outputs/qa/milestone_3/market_segmentation/cluster_selection_diagnostics.csv`
- `outputs/qa/milestone_3/market_segmentation/segmentation_manifest.json`

**Statistical governance / Gold**
- `outputs/tables/final_reporting/statistical_method_map.csv`
- `outputs/tables/final_reporting/robustness_sensitivity_matrix.csv`
- `outputs/tables/final_reporting/statistical_readiness_matrix.csv`
- `outputs/tables/final_reporting/statistical_design_summary.csv`
- `outputs/qa/final_reporting/statistical_governance_manifest.json`
- `data/gold/metadata/dashboard_quality_status.csv`
- `data/gold/qa/gold_qa_manifest.json`

This report does not refit statistical or machine-learning models; it consolidates accepted project artifacts and frozen methodological contracts.
