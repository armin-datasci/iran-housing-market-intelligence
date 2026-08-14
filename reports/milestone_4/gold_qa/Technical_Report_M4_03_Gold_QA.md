# Technical Report - M4 Gold Data Contract QA

- Version: `m4-gold-qa-v3.4-section29-status-location-handoff-hotfix`
- Overall status: `PASS`
- Critical failures: `0`
- Review checks: `0`
- Ready for M4-05 Measure/DAX design: `True`

## Frozen architecture

- 10 dashboard-facing marts and 5 conformed dimensions.
- Exact section-29 P01-P08 page registry.
- `mart_location_market` combines compatible Price Map and Market Temperature location grains.
- `mart_text_monthly` supplies month-sensitive text frequency; adjusted text estimates remain fixed in `mart_text_signals`.
- Segment profile and monthly mix carry segment ID/name directly; no `dim_segment` is persisted.
- `dim_user_type` is semantic-only; no fake inactive/bidirectional relationship is created.

## Interpretation constraints

- Asking prices are not verified transaction prices.
- Listing activity is not physical inventory, liquidity, or absorption.
- Market Temperature uses the frozen all-city ranking; four-city scope is presentation/§28 only.
- Driver, seller, and text effects are observational/fixed estimates, not causal effects.
- Segment fallback is presented as Market Types / Descriptive Typology.
- Exact coordinates are excluded from Gold.
