# IHMI FastAPI Deployment

This package exposes the accepted IHMI Gold layer through a read-only FastAPI service.

## Scope

The API:

- reads only canonical Gold Parquet artifacts;
- does not rebuild Silver or Gold;
- does not refit models;
- does not recompute Market Temperature;
- does not recalculate segmentation;
- does not expose exact listing coordinates;
- preserves asking-price, listing-activity, and non-causal interpretation boundaries.

## Local run

From the repository root:

```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
```

Open:

```text
http://127.0.0.1:8000/docs
http://127.0.0.1:8000/health
```

## Main endpoints

```text
GET /
GET /health
GET /api/v1/meta
GET /api/v1/datasets

GET /api/v1/market/monthly
GET /api/v1/market/locations

GET /api/v1/drivers/effects
GET /api/v1/drivers/importance
GET /api/v1/model/quality

GET /api/v1/seller

GET /api/v1/text/signals
GET /api/v1/text/monthly

GET /api/v1/segments
GET /api/v1/segments/monthly

GET /api/v1/dimensions/{dimension_name}
```

FastAPI automatically provides interactive OpenAPI/Swagger documentation at `/docs`.

## Stage deployment data

The hosted API needs the compact Gold Parquet artifacts. Create a privacy-checked deployment bundle:

```powershell
python .\scripts\export_api_bundle.py --reset
```

This creates:

```text
deployment/fastapi/api_data/gold/
|-- marts/
|-- dimensions/
|-- qa/
`-- api_bundle_manifest.json
```

The export command validates:

- the canonical mart/dimension allowlists;
- required schema fields;
- absence of forbidden coordinate/geometry/legacy columns.

## Docker test

After staging the API data:

```powershell
docker build -f deployment/fastapi/Dockerfile -t ihmi-fastapi .
docker run --rm -p 8000:8000 ihmi-fastapi
```

Then open:

```text
http://127.0.0.1:8000/docs
```

Run the API smoke test from another terminal:

```powershell
python .\scripts\smoke_test_api.py --base-url http://127.0.0.1:8000
```

## Environment variables

```text
IHMI_PROJECT_ROOT
IHMI_GOLD_DIR
IHMI_API_PREFIX
IHMI_API_CORS_ORIGINS
IHMI_API_DEFAULT_PAGE_SIZE
IHMI_API_MAX_PAGE_SIZE
PORT
```

For a separate browser frontend, set `IHMI_API_CORS_ORIGINS` to explicit comma-separated origins instead of using a wildcard.

## Deployment

The Docker image can be deployed to a container hosting platform after `api_data/gold` has been staged.

The service command is already defined in the Dockerfile:

```text
uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000}
```

A public deployment URL plus a successful `/health` response and accessible `/docs` page can be used as deployment evidence for the project bonus.

## Bonus evidence checklist

Before claiming the deployment bonus, verify:

```text
[ ] Public HTTPS URL works
[ ] GET /health returns status=ok
[ ] Gold QA status is visible
[ ] /docs is publicly accessible
[ ] At least one market endpoint returns data
[ ] No exact coordinates are exposed
[ ] Asking-price and listing-activity claim boundaries remain documented
[ ] Deployment URL is included in the final project README/report/demo
```
