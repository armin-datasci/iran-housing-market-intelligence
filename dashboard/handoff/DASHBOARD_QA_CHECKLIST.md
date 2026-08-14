# Dashboard QA Checklist

Use this checklist before the PBIX is accepted.

## Data source / model

- [ ] Power BI source parameter is relative/portable (`pProjectRoot`); no personal path is embedded in M code.
- [ ] Exactly 10 Gold mart Parquets are loaded.
- [ ] Exactly 5 Gold dimension Parquets are loaded.
- [ ] `dashboard_quality_status.csv` is loaded as disconnected metadata for P02.
- [ ] Exactly 13 physical relationships match `dashboard_relationship_contract.csv`.
- [ ] All 13 relationships are Active, Single, `1:*`.
- [ ] `dim_user_type` is disconnected.
- [ ] No `dim_segment` was created as a physical Gold dimension.
- [ ] No exact latitude/longitude/geometry is loaded.
- [ ] `dim_month[month_label]` sorts by `chronological_sort`.

## Semantics

- [ ] Asking price is never presented as transaction price.
- [ ] Toman is labeled as operational/source-unconfirmed where financial context is shown.
- [ ] Listing activity is not called inventory/liquidity/absorption.
- [ ] Market Temperature is explicitly a proxy.
- [ ] Adjusted price-driver effects are labeled association/contrast, not causal effect/share of price.
- [ ] Permutation importance is labeled predictive contribution.
- [ ] Seller adjusted difference is observational and fixed.
- [ ] Text adjusted effects do not change with Month slicer; only prevalence changes.
- [ ] Segments are labeled Market Types / Descriptive Typology; no false stable-cluster claim.
- [ ] N/reliability status is shown in visual/tooltips where required.

## Pages / filters

- [ ] Exactly P01-P08 page names are used.
- [ ] All seven mandatory filter controls exist in the report experience.
- [ ] Filter interactions match `dashboard_filter_contract.csv`.
- [ ] City/Neighborhood are Single-select or All for monthly median measures.
- [ ] Minimum Valid N cannot weaken canonical reliability.
- [ ] P03 Top-5 HOT/COLD uses neighborhood + four-city + canonical reliability.
- [ ] P05 contains AVM/error evidence in addition to price-driver effects.
- [ ] P02 does not pretend upstream QA is recalculated by analytical slicers.

## Reconciliation / delivery

- [ ] `dashboard_reference_metrics.csv` reconciles within tolerance.
- [ ] Gold QA manifest remains PASS/ready after refresh.
- [ ] Refresh works after moving the extracted handoff folder and changing only `pProjectRoot`.
- [ ] No secrets/tokens/personal paths exist in PBIX queries or documentation.
