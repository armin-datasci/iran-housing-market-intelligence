# Dashboard Semantic Notes

These notes are mandatory wording/interpretation constraints for the design team.

1. **Prices are asking prices, not transaction prices.** Currency is operationally Toman but remains source-unconfirmed. Do not silently apply a Rial/Toman factor.
2. **Listing activity is a platform-flow proxy**, not physical housing inventory, liquidity, absorption, or time-to-sale.
3. **Market Temperature** combines asking-price and listing-activity trend evidence. Label it `Listing-Market Temperature` or `Market Temperature Proxy`; never call it liquidity/absorption.
4. **HOT/COLD final presentation:** neighborhood-level, four-city view (`dim_location[is_four_city]=TRUE`) with the upstream reliability gate. The underlying analysis remains all-city.
5. **Price-driver effects** are model-implied adjusted associations/contrasts, not causal effects and not “share of price.” The correct visual title is `Adjusted Asking-Price Association by Property Characteristic`.
6. **Permutation importance** is held-out predictive contribution. Grouped Location and Property-Type blocks are structural controls and should be visually distinguished from actionable property characteristics.
7. **Seller Type** is one fixed agency-vs-personal observational adjusted contrast. The User Type slicer is disconnected and may only switch descriptive cards; it must not recompute the adjusted estimate.
8. **Text effects** are fixed controlled observational associations for six manually validated signals. Month selection applies only to prevalence/frequency, not to the fixed adjusted effect.
9. **Segments** are five `Market Types / Descriptive Typology` outputs. Do not label them stable clusters when the clustering gate did not pass. Keep property-applicability N/A states visible.
10. **Reliability is part of the result.** Show N/sample, PASS/REVIEW/reliability state, or an equivalent visual cue on every page where interpretation depends on it.
11. **Minimum Valid N is a display threshold**, not a way to weaken upstream canonical reliability. DAX gates use the stricter of user-selected N and the canonical threshold.
12. **No exact coordinates** belong in Power BI. Text geocoding or aggregate locations are allowed.
