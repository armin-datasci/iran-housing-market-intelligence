# Power BI Semantic Contracts

This folder is the canonical destination for **dashboard-specific** M4-05 artifacts. It does not replace `data/gold/metadata/`; it consumes the frozen Gold contract.

Expected files after M4-05 design is finalized:

```text
dashboard_measure_catalog.csv
dashboard_page_contract.csv
dashboard_filter_contract.csv
dashboard_relationship_contract.csv
```

Optional supporting files may include a DAX source/catalog and a concise dashboard handoff note, but duplicate copies of Gold marts/dimensions or task-level QA should not be stored here.

## Contract principles

- Every visual uses a predefined measure where a measure is required.
- Measures declare their source mart, unit/format, measure class, and valid filter behavior.
- Fixed analytical estimates must not appear to refit under unsupported slicers.
- Page names remain exactly P01-P08 from assignment section 29.
- Physical relationships must reconcile with the frozen 13-row Gold relationship contract.
- `dim_user_type` remains semantic-only; no fake physical relationship is created.
- Minimum valid listing count is a disconnected analytical/display parameter rather than a business dimension.
