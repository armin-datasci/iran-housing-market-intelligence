# FastAPI Public Deployment Evidence

## Deployment

- **Service:** Iran Housing Market Intelligence API
- **Provider:** Render
- **Public base URL:** https://ihmi-fastapi.onrender.com
- **Interactive Swagger/OpenAPI:** https://ihmi-fastapi.onrender.com/docs
- **Health endpoint:** https://ihmi-fastapi.onrender.com/health
- **GitHub repository:** https://github.com/armin-datasci/iran-housing-market-intelligence
- **Validation date:** 2026-08-14

## Verified public checks

```text
IHMI FASTAPI SMOKE TEST: PASS
health: ok
gold_qa_status: PASS
marts: 10
dimensions: 5
Swagger /docs: HTTP 200
```

Representative public data endpoints were also verified with HTTP 200 responses, including:

```text
GET /api/v1/market/monthly?limit=1
GET /api/v1/drivers/effects?limit=1
GET /api/v1/model/quality?limit=1
GET /api/v1/meta
```

The public health response confirmed:

```text
status = ok
gold_qa_status = PASS
expected_marts = 10
expected_dimensions = 5
missing_artifacts = []
schema_issues = {}
forbidden_columns = {}
```

## Scope and interpretation boundaries

The deployed API is read-only and consumes accepted canonical Gold artifacts. It does not rebuild Silver or Gold, refit models, or recompute Market Temperature or segmentation.

The deployment preserves the project interpretation contract:

- asking prices are not verified transaction prices;
- listing activity is platform activity, not physical inventory, liquidity, or absorption;
- Market Temperature is a Listing-Market Temperature Proxy;
- price-driver, seller-type, and text results are observational/predictive rather than causal;
- AVM outputs are research/prototype diagnostics, not production valuation guarantees;
- market segments are descriptive market types, not guaranteed latent clusters;
- exact listing coordinates are not exposed through the API.

## Reproducibility

Local/public smoke test command:

```powershell
python .\scripts\smoke_test_api.py --base-url https://ihmi-fastapi.onrender.com
```

Swagger availability check:

```powershell
curl.exe -sS -o NUL -w "HTTP=%{http_code}`n" "https://ihmi-fastapi.onrender.com/docs"
```
