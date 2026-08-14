# IHMI Read-Only FastAPI

## Purpose

The FastAPI service is a deployment/access layer over the canonical Gold outputs. It is intentionally separated from analytical computation.

## Data contract

The API imports the Milestone 4 Gold structural contract directly from:

```text
src/milestone_4/gold/contracts.py
```

This keeps the API aligned with the canonical:

```text
10 marts
5 dimensions
13 physical relationships
```

The service only allows datasets declared in `EXPECTED_MARTS` and `EXPECTED_DIMENSIONS`.

## Privacy

At application validation time and during deployment-data export, columns containing forbidden spatial or legacy tokens are rejected. The API therefore does not expose exact coordinates or geometry.

## Claim boundaries

API consumers must preserve the following meanings:

- asking prices are not transaction prices;
- listing activity is not inventory, liquidity, or absorption;
- Market Temperature is a listing-market proxy;
- driver, seller, and text results are non-causal;
- AVM outputs are research/prototype diagnostics;
- market segmentation is a descriptive typology.

## OpenAPI

FastAPI provides:

```text
/docs
/redoc
/openapi.json
```

These endpoints provide a convenient live demonstration of the deployed API contract.
