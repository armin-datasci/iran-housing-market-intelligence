# Iran Housing Market Intelligence (IHMI)

## Overview

Iran Housing Market Intelligence (IHMI) is an end-to-end analytics project for transforming real-estate listing data into a governed analytical data layer, reproducible statistical evidence, management-ready Gold marts, a Power BI semantic layer, final reports, and a lightweight presentation notebook.

The project is designed around four analytical pipelines that map directly to the four project milestones.

> **Interpretation boundary:** all monetary values represent **listing asking prices**, not verified transaction prices. Listing activity represents observed platform activity and must not be interpreted as physical housing inventory, liquidity, or absorption.

---

## Project Architecture

The persisted analytical architecture contains two data layers:

```text
External Raw Source
        ↓
Silver Master
        ↓
Gold
        ↓
Dashboard / Reports / Final Notebook
```

Canonical paths:

```text
external_data/real_estate_ads.csv
data/silver/silver_master.parquet
data/gold/
```

The Silver Master is the canonical analytical dataset. Gold contains compact reporting and dashboard-ready marts, dimensions, metadata, and QA artifacts derived from accepted Silver and analytical outputs.

---

## Pipeline 1 — Milestone 1: Data Understanding and Documentation

Pipeline 1 establishes the dataset structure, documentation, and initial quality evidence.

Main outputs:

- Data Dictionary
- Data Contract
- Raw Data Quality Audit
- Milestone 1 validation and closeout evidence

The canonical Data Dictionary and Data Contract document the project schema and derived analytical fields while ensuring complete coverage of the raw source schema.

---

## Pipeline 2 — Milestone 2: Data Cleaning and Silver Master

Pipeline 2 applies the complete data-quality and cleaning workflow before producing the Silver Master.

Execution order:

```text
Persian/Arabic character and digit standardization
        ↓
Missing-value management
        ↓
Duplicate control and Advanced Entity Resolution
        ↓
Group-aware outlier policy and sensitivity analysis
        ↓
Currency decision
        ↓
Canonical price regimes
        ↓
Deposit-to-rent conversion sensitivity
        ↓
Final price metrics
        ↓
Data Quality Gate
        ↓
Silver Master
```

### Standardization

The cleaning layer normalizes Persian/Arabic characters, digits, structured categorical fields, and typed numeric/date fields while preserving auditability.

### Missing Values

Missingness is handled by field-specific policy. Null values are preserved when they represent unknown, unavailable, or non-applicable information. Blanket conversion of missing values to zero or `False` is avoided.

### Duplicate Control and Advanced Entity Resolution

Duplicate management distinguishes between:

- exact duplicates,
- high-confidence probable duplicates,
- medium-confidence probable duplicates,
- cross-month repeated listings.

Only sufficiently reliable same-month duplicate excess can affect conservative supply eligibility. Cross-month repetition remains auditable because repeated listings may represent continued platform activity.

### Outlier Policy

The canonical outlier rule is group-aware and flag-only.

```text
Canonical rule: 3 × IQR
Sensitivity rule: 2 × IQR
```

Outliers are retained in Silver and represented through quality flags rather than destructive row deletion.

### Currency

The operational currency status is:

```text
toman_assumed_unconfirmed
```

No automatic 10× currency conversion is applied without source confirmation. Currency uncertainty is retained explicitly in QA and reporting.

### Price Regimes

Sale, rent, deposit, and mixed monetary observations are separated through canonical price-regime logic so incompatible price concepts are not combined.

### Silver Master

Canonical output:

```text
data/silver/silver_master.parquet
```

The Silver Master is read-only for downstream analytical pipelines.

---

## Pipeline 3 — Milestone 3: Market Intelligence and Modeling

Pipeline 3 produces the analytical evidence used by reports, Gold, the dashboard, and the final presentation notebook.

Main analytical sequence:

```text
Analysis Populations
        ↓
Spatial Validation and Advanced Spatial Analysis
        ↓
Aggregate Market Map
        ↓
Monthly Listing Activity and Asking-Price Trends
        ↓
Listing-Market Temperature
        ↓
Price Drivers / Interpretable AVM / AVM Error Analysis
        ↓
Seller-Type Comparison
        ↓
Precision-Validated Text Signals
        ↓
Market Segmentation / Descriptive Market Typology
```

### Analysis Populations

Separate analytical populations are declared for sale, rent, supply, mapping, and model-specific tasks. Downstream analyses use compatible populations rather than mixing incompatible price regimes or analytical grains.

### Spatial Validation and Advanced Spatial Analysis

Spatial QA includes:

- coordinate validity and coverage checks,
- city-level spatial consistency checks,
- external ADM2 boundary point-in-polygon validation,
- stratified and sanitized reverse-geocode validation,
- coordinate-reuse and suspicious-location checks,
- publication controls that exclude exact listing coordinates from Gold and the dashboard.

### Interactive Market Map

Canonical map artifact:

```text
outputs/maps/milestone_3/market_map/four_city_market_map.html
```

The final notebook provides a lightweight hyperlink to this file rather than embedding the complete HTML document inline.

### Monthly Market Analysis

Monthly analysis reports platform listing activity and apartment-sale asking-price trends over the accepted core period.

Listing activity is interpreted as platform activity, not market inventory or liquidity.

### Listing-Market Temperature

Market Temperature is a relative **Listing-Market Temperature Proxy** based on accepted price and activity evidence.

It must not be described as liquidity, absorption, transaction velocity, or market depth.

### Price Drivers and Interpretable AVM

The price-driver workflow uses a controlled Ridge model with held-out evaluation.

Core outputs include:

```text
outputs/tables/milestone_3/price_drivers/price_driver_summary.csv
outputs/tables/milestone_3/price_drivers/price_driver_model_diagnostics.csv
outputs/tables/milestone_3/price_drivers/price_model_benchmark.csv
outputs/tables/milestone_3/price_drivers/price_driver_permutation_importance.csv
outputs/tables/milestone_3/price_drivers/avm_error_analysis.csv
```

The model supports:

- adjusted asking-price associations,
- held-out predictive contribution,
- interpretable AVM diagnostics,
- AVM error analysis by meaningful market segments.

The AVM is a research/prototype predictive model and is not a production valuation guarantee.

### Seller-Type Comparison

Agency and personal listings are compared using:

- raw asking-price evidence,
- controlled cross-fitted residual comparison,
- similar-unit/coarsened-strata robustness.

The adjusted result is observational and does not establish a causal agency premium.

### Text Signals

Text analysis uses manually precision-validated keyword families.

Validated text signals are evaluated using:

- held-out control-only residuals,
- Welch inference,
- Benjamini-Hochberg false-discovery-rate control.

Text results are observational associations and must not be interpreted as causal seller intent.

### Market Segmentation

The released segmentation is a **descriptive market typology**.

Unsupervised clustering candidates are evaluated using predefined quality and stability diagnostics, including:

- Silhouette,
- Calinski-Harabasz,
- Davies-Bouldin,
- minimum cluster share,
- resampling ARI.

The final released segmentation is presented as descriptive market types rather than latent statistical classes.

---

## Pipeline 4 — Milestone 4: Gold Layer and Dashboard Handoff

Pipeline 4 converts accepted analytical outputs into a compact semantic layer for reporting and Power BI.

Final Gold contract:

```text
10 marts
5 dimensions
13 active single-direction physical relationships
```

Gold directories:

```text
data/gold/marts/
data/gold/dimensions/
data/gold/metadata/
data/gold/qa/
```

Gold contains only reporting-ready metrics, dimensions, quality metadata, and accepted analytical summaries. Exact listing coordinates are excluded.

`dim_user_type` is used as a disconnected semantic dimension where appropriate and is not forced into a misleading physical relationship.

The dashboard consumes Gold outputs and must not refit analytical models or recompute fixed model estimates through slicers.

---

## Statistical Governance

Canonical statistical governance module:

```text
src/final_reporting/statistical_governance.py
```

Canonical governance artifacts are stored under:

```text
outputs/tables/final_reporting/
outputs/qa/final_reporting/
```

The final presentation notebook surfaces two governance tables:

### 1. Model / Statistical-Method Summary

This table summarizes the main accepted modeling methods and their key evaluation metrics:

- Controlled Ridge AVM / Price Drivers
- Cross-fitted control-only Ridge for Seller-Type Comparison
- Held-out Ridge residual analysis for validated Text Signals

### 2. Statistical Design Summary

This table summarizes:

- analytical populations,
- unit of analysis,
- major biases,
- missing-value policy,
- outlier policy,
- uncertainty and inference,
- sensitivity checks,
- leakage controls,
- claim boundaries.

No model is fitted inside the final notebook.

---

## Final Notebook

Canonical notebook:

```text
notebooks/final_analysis.ipynb
```

The final notebook is a lightweight evidence and presentation layer.

It:

- loads accepted canonical artifacts,
- displays the main tables and figures,
- provides short interpretations,
- presents the project as four milestone-aligned pipelines,
- does not rebuild Silver or Gold,
- does not rerun the complete canonical pipeline,
- does not rerun the automated test suite,
- does not refit analytical models.

Notebook structure:

```text
Pipeline 1 → Milestone 1
Pipeline 2 → Milestone 2
Pipeline 3 → Milestone 3
Pipeline 4 → Milestone 4
        ↓
Statistical Governance
```

---

## Running the Canonical Pipeline

From the repository root:

```powershell
.\.venv\Scripts\Activate.ps1
python .\scriptsun_pipeline.py --start-at m1 --stop-after reports
```

Canonical orchestrator:

```text
scripts/run_pipeline.py
```

The final reporting stage runs:

```text
Environment Snapshot
        ↓
Statistical Governance
        ↓
Executive Summary
        ↓
Technical Report
```

Important QA artifacts:

```text
outputs/qa/final_pipeline/environment_versions.json
outputs/qa/final_pipeline/final_pipeline_manifest.json
outputs/qa/final_reporting/statistical_governance_manifest.json
```

Final professor-facing reports:

```text
reports/final/Executive_Summary.md
reports/final/Technical_Report.md
```

---

## Running the Final Notebook

Start JupyterLab from the repository root:

```powershell
.\.venv\Scripts\Activate.ps1
python -m jupyter lab
```

Open:

```text
notebooks/final_analysis.ipynb
```

Recommended kernel:

```text
Python (IHMI)
```

Verify the interpreter if needed:

```python
import sys
print(sys.executable)
```

Then execute:

```text
Kernel
→ Restart Kernel and Run All Cells
```

JupyterLab should be launched from the repository root so the local hyperlink to the interactive market map can be served correctly.

---

## Repository Structure

```text
iran-housing-market-intelligence/
├── config/
├── docs/
├── external_data/
│   ├── real_estate_ads.csv
│   └── reference/
├── data/
│   ├── silver/
│   │   └── silver_master.parquet
│   └── gold/
│       ├── marts/
│       ├── dimensions/
│       ├── metadata/
│       └── qa/
├── src/
│   ├── common/
│   ├── milestone_1/
│   ├── milestone_2/
│   ├── milestone_3/
│   ├── milestone_4/
│   └── final_reporting/
├── scripts/
│   └── run_pipeline.py
├── notebooks/
│   └── final_analysis.ipynb
├── outputs/
│   ├── tables/
│   ├── model_artifacts/
│   ├── figures/
│   ├── maps/
│   └── qa/
├── dashboard/
│   ├── contracts/
│   └── handoff/
├── reports/
│   └── final/
│       ├── Executive_Summary.md
│       └── Technical_Report.md
├── tests/
├── decision_log.md
├── requirements.txt
└── README.md
```

---

## Environment and Reproducibility

Install dependencies using:

```powershell
python -m pip install -r requirements.txt
```

The runtime environment is captured in:

```text
outputs/qa/final_pipeline/environment_versions.json
```

The environment snapshot records package/runtime information required for reproducibility without storing credentials, personal paths, or environment secrets.

---

## Validation Status

Current technical validation:

```text
Python compileall      PASS
Automated tests        47 passed
Static/path scan       PASS
Secret scan            PASS
Credential-file scan   PASS
Canonical pipeline     PASS
Gold QA                PASS
```

Milestone 2 may retain a `REVIEW` status while remaining analytically ready because the operational Toman currency interpretation is source-unconfirmed. This state is explicitly preserved in the project quality contract.

---

## Interpretation and Claim Boundaries

The following rules apply across Python outputs, the final notebook, reports, Gold, and the dashboard:

- Asking price is not transaction price.
- Listing activity is not physical housing inventory.
- Listing activity is not liquidity or absorption.
- Market Temperature is a listing-market proxy.
- Price-driver effects are adjusted observational associations.
- Permutation importance represents held-out predictive contribution.
- Seller-type differences are observational associations.
- Text-price signals are validated observational associations.
- AVM outputs are research/prototype predictive diagnostics.
- Market segments are descriptive market types.
- Exact listing coordinates must not appear in Gold or the dashboard.

---

## Dashboard

The Power BI dashboard is built on the canonical Gold layer.

Semantic and handoff material is stored under:

```text
data/gold/metadata/
dashboard/contracts/
dashboard/handoff/
```

Dashboard filters must respect the applicability of each upstream artifact. Fixed analytical estimates must not be implicitly refitted or recomputed by dashboard filtering.

---

## Decision Log

Project decisions are documented in:

```text
decision_log.md
```

The decision log contains the canonical decisions covering:

- currency and asking-price interpretation,
- persisted data architecture,
- price regimes,
- duplicate policy,
- outlier policy,
- spatial QA,
- analytical scope and Market Temperature,
- non-causal modeling interpretation,
- market-segmentation release logic,
- Gold/dashboard contract,
- pipeline, notebook, and reporting architecture.

---

## Final Deliverables

The final project package contains:

```text
1. Executive Summary
2. Power BI Dashboard
3. Technical Report
4. Final Restart-and-Run-All Notebook
5. Product Recommendations
6. Demo / Presentation
```

Professor-facing written reports:

```text
reports/final/Executive_Summary.md
reports/final/Technical_Report.md
```

Task-level QA tables, manifests, model diagnostics, and validation artifacts are retained for auditability and technical defense.

---

## Security Rules

- Do not commit secrets, passwords, API keys, or tokens.
- Do not commit `.env`, private keys, or credential files.
- Do not hard-code personal filesystem paths.
- Do not publish exact listing coordinates in Gold or the dashboard.
- Run the canonical pipeline from the repository root.
- Keep analytical logic inside canonical source modules rather than duplicating it in the final notebook or Power BI.
