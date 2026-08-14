# IHMI Test Strategy — Consolidated Final Suite

The final project keeps **8 test files** rather than many one-function files. The goal is minimum file fragmentation with strong coverage of the contracts that can materially change rubric results or downstream reproducibility.

| Suite | Coverage |
|---|---|
| `milestone_1/test_m1_contract.py` | Raw inspection, Data Dictionary/Data Contract, M1 closeout surface |
| `milestone_2/test_cleaning_contract.py` | Persian/Arabic + structured standardization, missingness, currency, 9 price regimes, Technical Report standardization evidence |
| `milestone_2/test_quality_metrics_contract.py` | Duplicate/Advanced ER, final price metrics, outlier v2 configuration, quality gate, M2 closeout |
| `milestone_3/test_market_spatial_contract.py` | M2→M3 schema, map privacy, monthly/temperature closeout, reliability gate, Advanced Spatial bonus |
| `milestone_3/test_model_text_contract.py` | Price-driver/AVM behavior, error analysis, seller comparison, text rules, FDR |
| `milestone_3/test_segmentation_contract.py` | Local price reference, fallback typology, compatible family segments, semantic labels |
| `milestone_4/test_gold_contract.py` | Frozen 10 marts + 5 dimensions + 13 relationships, adapters, grains, page registry, semantic safety |
| `final/test_delivery_contract.py` | Single orchestrator, final report outputs, dashboard contract surface, notebook presence/JSON validity |

This consolidation changes **file organization**, not the analytical methodology. Important regression assertions from the previous suite are retained; exact duplicate assertions are removed. Heavy end-to-end data validation remains owned by the canonical M1–M4 QA manifests and `scripts/run_pipeline.py`/the current single runner during the architecture transition.
