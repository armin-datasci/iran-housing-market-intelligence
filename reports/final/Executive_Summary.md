# Executive Summary - Iran Housing Market Intelligence

- **Report version:** `final-executive-summary-v2.3-public-fastapi-deployment`
- **Generated at (UTC):** `2026-08-14T01:30:25.620849+00:00`
- **Core analytical period:** `2024-05` to `2024-12` (8 months)
- **Price observation:** listing asking price, not transaction price
- **Operational currency:** Toman (`toman_assumed_unconfirmed`)
- **Gold status:** Gold QA=PASS; critical failures=0; data contract ready=True; architecture=10 marts/5 dimensions/13 physical relationships.

## 1. Overall market status

The final analysis uses the canonical Silver Master, accepted analytical outputs, and the validated Gold data layer. In the national apartment-sale series, deduplicated listing activity increased from **34,949** listings in `2024-05` to **41,328** in `2024-12`, a cumulative change of **+18.3%**. The median valid month-over-month change in deduplicated listing activity was **+0.59%**.

Over the same period, the national median apartment asking price per square meter moved from **36,363,636 Toman** to **36,538,462 Toman**, a cumulative change of **+0.5%**. The median valid month-over-month asking-price change was **+0.02%**. These values describe listing behavior, not completed transactions.

Market Temperature is a **Listing-Market Temperature Proxy** based on asking-price trend and listing-activity trend. The canonical ranking remains all-city within each entity level, while the executive top/bottom tables reproduce the project's professor-facing reliable neighborhood view for **Isfahan, Karaj, Mashhad, Tehran**. The reliability gate is at least **5 price months**, **5 listing-activity months**, and **N >= 100**. In this presentation pool there are **17 HOT**, **30 COLD**, and **98 NEUTRAL** neighborhoods.

## 2. Five hottest neighborhood markets

| Market | Temperature score | Asking-price trend / month | Listing-activity trend / month | N |
|---|---|---|---|---|
| Karaj / Golshahrvila | 75.9 | +1.37% | +8.10% | 327 |
| Mashhad / Daneshjoo | 70.0 | +1.24% | +7.09% | 364 |
| Isfahan / Eshragh | 66.3 | +0.99% | +7.97% | 476 |
| Tehran / Zahir Abad | 63.7 | +1.22% | +5.70% | 162 |
| Tehran / Seyed Khandan | 62.8 | +1.09% | +5.85% | 276 |

HOT indicates a positive relative proxy signal after the reliability gate. It does not mean highest liquidity, best investment return, or strongest physical demand.

## 3. Five coldest neighborhood markets

| Market | Temperature score | Asking-price trend / month | Listing-activity trend / month | N |
|---|---|---|---|---|
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
|---|---|---|
| Location controls (City + Neighborhood; grouped) | location control | 0.2701 |
| Rooms | property characteristic | 0.0477 |
| Building age | property characteristic | 0.0389 |
| Primary area | property characteristic | 0.0339 |
| Property-type controls (Family + Category; grouped) | property-type control | 0.0115 |
| Rebuilt | property characteristic | 0.0099 |

Permutation importance measures **held-out predictive contribution**, not causal importance. Location and property-type blocks are structural controls and should be separated from potentially actionable property characteristics.

### 5.2 Model-implied adjusted associations

| Property characteristic | Contrast | Adjusted asking-price association |
|---|---|---|
| building direction:east | east vs modal reference south | -12.0% |
| Primary area | p75 (160) vs p25 (75) | -8.1% |
| building direction:west | west vs modal reference south | -6.4% |
| building direction:north | north vs modal reference south | -3.1% |
| Building age | p75 (13) vs p25 (1) | +2.8% |

Adjusted effects are **model-implied associations/contrasts**. They are not shares of price and they are not causal effects.

## 6. Main limitations

1. **Asking-price data:** observed prices are listing asks, not realized transaction prices.
2. **Currency status:** Toman is the project's operational unit, but the source currency is not independently confirmed; no undocumented factor-of-ten conversion is applied.
3. **Platform selection bias:** the dataset is not the complete Iranian housing stock or transaction universe.
4. **Market Temperature semantics:** the index combines asking-price and listing-activity trends; it is not liquidity, absorption, physical inventory, or supply tightness.
5. **Model interpretation:** price-driver results are observational and non-causal. On the held-out test set, the Ridge model had R2(log)=0.277 and median absolute percentage error=33.6%; it is therefore more appropriate for approximate prediction and association analysis than point-precise valuation.
6. **Spatial privacy:** exact coordinates are not exposed in Gold or the dashboard; spatial outputs are aggregated.
7. **Presentation scope:** the five HOT/COLD tables are the reliable four-city professor-facing view, while the underlying temperature model and canonical labels are computed on the broader eligible ranking universe.

## 7. Management takeaway

The core-period listing market is heterogeneous rather than uniformly hot or cold. National apartment-sale listing activity increased over the period, while the national median asking price per square meter changed much less. Local differences are substantial, so HOT/COLD rankings should be read together with sample size, asking-price trend, listing-activity trend, and reliability conditions. Predictive modeling confirms that location and structural property characteristics matter for prediction, but the evidence does not by itself establish causal price effects or investment recommendations.

As a deployment bonus, the accepted Gold data products are also available through a public, read-only FastAPI service at `https://ihmi-fastapi.onrender.com`, with interactive Swagger documentation at `https://ihmi-fastapi.onrender.com/docs`. Public validation on **2026-08-14** returned `health=ok`, `gold_qa_status=PASS`, `marts=10`, `dimensions=5`, and HTTP `200` for `/docs`. The API does not expose exact listing coordinates and preserves the same interpretation boundaries as the analytical reports and dashboard.

---

### Canonical sources used

- `temperature_selection_gold`: `data/gold/marts/mart_location_market.parquet`
- `temperature_source_reconciliation`: `outputs/tables/milestone_3/market_temperature/market_temperature_summary.csv`
- `dim_location`: `data/gold/dimensions/dim_location.parquet`
- `temperature_gate`: `outputs/qa/milestone_3/market_temperature/market_temperature_manifest.json`
- `monthly_market`: `data/gold/marts/mart_market_monthly.parquet`
- `price_driver_importance`: `data/gold/marts/mart_price_driver_importance.parquet`
- `price_driver_effects`: `data/gold/marts/mart_price_driver_effects.parquet`
- `model_quality`: `data/gold/marts/mart_model_quality.parquet`
- `gold_qa`: `data/gold/qa/gold_qa_manifest.json`

This report builder does not refit models or recompute upstream analytical estimates; it summarizes accepted canonical outputs.
