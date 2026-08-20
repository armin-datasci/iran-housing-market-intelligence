# Executive Summary - Iran Housing Market Intelligence

- **Report version:** `final-executive-summary-v2.4-stage2-complete`
- **Analytical snapshot:** accepted pipeline/Gold evidence through `2026-08-14`
- **Documentation revision:** `2026-08-20`
- **Core analytical period:** `2024-05` to `2024-12` (8 months)
- **Price observation:** listing asking price, not transaction price
- **Operational currency:** Toman (`toman_assumed_unconfirmed`)
- **Gold status:** Gold QA=`PASS`; critical failures=`0`; data contract ready=`True`; architecture=`10 marts / 5 dimensions / 13 physical relationships`

## 1. Overall market status

The final analysis uses the canonical Silver Master, accepted analytical outputs, and the validated Gold data layer. In the national apartment-sale series, deduplicated listing activity increased from **34,949** listings in `2024-05` to **41,328** in `2024-12`, a cumulative change of **+18.3%**. The median valid month-over-month change in deduplicated listing activity was **+0.59%**.

Over the same period, the national median apartment asking price per square meter moved from **36,363,636 Toman** to **36,538,462 Toman**, a cumulative change of **+0.5%**. The median valid month-over-month asking-price change was **+0.02%**. These values describe listing behavior, not completed transactions.

Market Temperature is a **Listing-Market Temperature Proxy** based on asking-price trend and listing-activity trend. The canonical ranking is computed across the broader eligible all-city universe within each entity level. The executive top/bottom tables reproduce the professor-facing reliable neighborhood view for **Isfahan, Karaj, Mashhad, and Tehran**. The reliability gate requires at least **5 price months**, **5 listing-activity months**, and **N >= 100**. In this presentation pool there are **17 HOT**, **30 COLD**, and **98 NEUTRAL** neighborhoods.

### Monetary-scale confidence

The project does not rely on an undocumented Rial/Toman conversion. The full currency audit covered **1,000,000 rows**, found **0 raw-to-typed mismatches across 11 monetary field pairs**, and found no material factor-of-ten subpopulation. Independent economic-coherence checks also recovered the rent/deposit structure and showed strong sale-versus-rent-equivalent geographic alignment: **25,400** rent-credit transformations centered on a **0.030** monthly-equivalent ratio, while **603** sufficiently populated neighborhoods produced a **0.9630** log-correlation between sale asking PSM and the base rent-equivalent PSM. These findings support keeping the supplied numerical scale unchanged (`scale=1`). They do **not** independently prove the absolute Rial/Toman denomination, so the project uses Toman operationally while preserving `toman_assumed_unconfirmed` as the formal source-status label.

## 2. Five hottest neighborhood markets

| Market | Temperature score | Asking-price trend / month | Listing-activity trend / month | N |
|---|---:|---:|---:|---:|
| Karaj / Golshahrvila | 75.9 | +1.37% | +8.10% | 327 |
| Mashhad / Daneshjoo | 70.0 | +1.24% | +7.09% | 364 |
| Isfahan / Eshragh | 66.3 | +0.99% | +7.97% | 476 |
| Tehran / Zahir Abad | 63.7 | +1.22% | +5.70% | 162 |
| Tehran / Seyed Khandan | 62.8 | +1.09% | +5.85% | 276 |

HOT indicates a positive relative proxy signal after the reliability and direction gates. It does not mean highest liquidity, best investment return, or strongest physical demand.

## 3. Five coldest neighborhood markets

| Market | Temperature score | Asking-price trend / month | Listing-activity trend / month | N |
|---|---:|---:|---:|---:|
| Tehran / Parastar | -89.0 | -2.32% | -6.25% | 177 |
| Karaj / Andishehnewcity | -88.4 | -2.61% | -3.39% | 1,954 |
| Tehran / North Sohrevardi | -85.2 | -1.71% | -5.78% | 565 |
| Karaj / Mehrshar 1 | -82.8 | -1.69% | -3.66% | 349 |
| Tehran / Pirouzi | -80.4 | -1.16% | -5.41% | 581 |

COLD indicates a negative relative proxy signal in the listing data. It should not be interpreted by itself as structural market weakness.

## 4. Supply and asking-price trends

- National deduplicated apartment-sale listing activity over the core period: **+18.3%**.
- National median apartment asking price per square meter over the core period: **+0.5%**.
- Median valid month-over-month listing-activity change: **+0.59%**.
- Median valid month-over-month asking-price change: **+0.02%**.
- Listing counts are a platform-activity measure; they are not physical housing inventory or transaction volume.

## 5. Most important factors associated with asking price

### 5.1 Held-out predictive contribution

| Feature / block | Role | Permutation increase in RMSE(log) |
|---|---|---:|
| Location controls (City + Neighborhood; grouped) | location control | 0.2701 |
| Rooms | property characteristic | 0.0477 |
| Building age | property characteristic | 0.0389 |
| Primary area | property characteristic | 0.0339 |
| Property-type controls (Family + Category; grouped) | property-type control | 0.0115 |
| Rebuilt | property characteristic | 0.0099 |

Permutation importance measures **held-out predictive contribution**, not causal importance. Location and property-type blocks are structural controls and should be separated from potentially actionable property characteristics.

### 5.2 Model-implied adjusted associations

| Property characteristic | Contrast | Adjusted asking-price association |
|---|---|---:|
| building direction:east | east vs modal reference south | -12.0% |
| Primary area | p75 (160) vs p25 (75) | -8.1% |
| building direction:west | west vs modal reference south | -6.4% |
| building direction:north | north vs modal reference south | -3.1% |
| Building age | p75 (13) vs p25 (1) | +2.8% |

Adjusted effects are **model-implied associations/contrasts**. They are not shares of price and they are not causal effects.

## 6. Additional analytical findings

### 6.1 Seller type: composition explains much of the raw gap

The accepted seller comparison covers **125,893** sale listings: **114,843 agency** and **11,050 personal** listings. The raw agency-versus-personal median asking-price gap is approximately **+92.0%**. After observed controls and cross-fitted comparison it falls to **+28.5%** with an approximate **95% interval of +26.0% to +31.0%**; a similar-unit/coarsened-strata sensitivity estimate is **+17.7%**. The shrinkage demonstrates the importance of composition and confounding. The remaining difference is observational and is **not** a causal agency premium.

### 6.2 Precision-validated text signals

Keyword families enter controlled price analysis only after manual precision validation. In the accepted BH-FDR-controlled analysis, `new_build` (**+13.1%**), `unused` (**+9.4%**), and `exchange` (**-11.0%**) retain statistically supported adjusted associations. `urgent` (**-6.5%**), `below_market` (**-4.3%**), and `migration_sale` (**+0.6%**) do not remain FDR-significant. These are observational associations in listing language, not causal seller-intent effects.

### 6.3 Market segmentation

The released five-group segmentation is a **Market Types / Descriptive Typology**. Unsupervised clustering candidates were evaluated with predeclared separation, minimum-cluster-share, and resampling-stability diagnostics, but unstable clustering was not forced into production. The final release therefore prioritizes compatibility and interpretability rather than claiming natural latent market classes.

## 7. Main limitations

1. **Asking-price data:** observed prices are listing asks, not realized transaction prices.
2. **Currency status:** internal evidence supports the supplied numerical scale, but the source denomination is not independently confirmed; Toman remains operational and no undocumented factor-of-ten conversion is applied.
3. **Platform selection bias:** the dataset is not the complete Iranian housing stock or transaction universe.
4. **Market Temperature semantics:** the index combines asking-price and listing-activity trends; it is not liquidity, absorption, physical inventory, or supply tightness.
5. **Model interpretation:** price-driver, seller-type, and text results are observational/predictive rather than causal. The held-out Ridge AVM has **R2(log)=0.277**, **Median APE=33.6%**, **P90 APE=87.8%**, and **44.7%** of test predictions within +/-30%; it is therefore an interpretable research prototype, not a point-precise production valuation system.
6. **Spatial privacy:** exact coordinates are not exposed in Gold, the dashboard, or the API; spatial outputs are aggregated.
7. **Presentation scope:** the five HOT/COLD tables are the reliable four-city professor-facing view, while the underlying temperature model and canonical labels are computed on a broader eligible ranking universe.
8. **Segmentation semantics:** released groups are descriptive market types, not socioeconomic classes, buyer personas, or guaranteed latent clusters.

## 8. Management takeaway

The core-period listing market is heterogeneous rather than uniformly hot or cold. National apartment-sale listing activity increased over the period, while the national median asking price per square meter changed much less. Local differences are substantial, so HOT/COLD rankings should be read together with sample size, asking-price trend, listing-activity trend, and reliability conditions.

The project also shows why raw descriptive differences should not be over-interpreted. Location dominates predictive performance, the seller-type gap shrinks materially after controls, and several intuitive text signals do not survive multiple-testing correction. The combined evidence supports a decision-support interpretation: use the platform data to compare relative listing-market conditions and model associations, while preserving explicit uncertainty, selection-bias, and causality boundaries.

As a deployment bonus, the accepted Gold data products are available through a public, read-only FastAPI service at `https://ihmi-fastapi.onrender.com`, with Swagger documentation at `https://ihmi-fastapi.onrender.com/docs`. The deployment consumes accepted Gold only; it does not rebuild Silver/Gold or refit analytical models, and it does not expose exact listing coordinates.

---

### Canonical sources used

- `data/silver/silver_master.parquet`
- `outputs/tables/milestone_2/currency/currency_validation_summary.csv`
- `outputs/tables/milestone_2/currency/currency_inference_interpretation.md`
- `data/gold/marts/mart_market_monthly.parquet`
- `data/gold/marts/mart_location_market.parquet`
- `data/gold/marts/mart_price_driver_importance.parquet`
- `data/gold/marts/mart_price_driver_effects.parquet`
- `data/gold/marts/mart_model_quality.parquet`
- `outputs/tables/milestone_3/seller_type_comparison/seller_type_comparison_summary.csv`
- `outputs/tables/milestone_3/seller_type_comparison/seller_type_stratified_summary.csv`
- `outputs/tables/milestone_3/text_price_signals/text_signal_summary.csv`
- `outputs/tables/milestone_3/market_segmentation/segment_profile.csv`
- `data/gold/qa/gold_qa_manifest.json`

This Executive Summary does not refit models or recompute upstream analytical estimates; it summarizes accepted canonical outputs.
