from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware

from src.api import API_VERSION
from src.api.config import load_settings
from src.api.repository import (
    DatasetContractError,
    GoldRepository,
    GoldRepositoryError,
    UnknownDatasetError,
)
from src.api.schemas import DatasetInfo, DatasetResponse, HealthResponse
from src.milestone_4.gold.contracts import EXPECTED_DIMENSIONS, EXPECTED_MARTS


SERVICE_NAME = "Iran Housing Market Intelligence API"
CLAIM_BOUNDARIES = [
    "Asking prices are not verified transaction prices.",
    "Listing activity is platform activity, not physical inventory, liquidity, or absorption.",
    "Market Temperature is a Listing-Market Temperature Proxy.",
    "Price-driver, seller-type, and text results are observational/predictive, not causal.",
    "AVM outputs are research/prototype diagnostics, not production valuation guarantees.",
    "Market segments are descriptive market types, not guaranteed latent clusters.",
    "Exact listing coordinates are not exposed by the API.",
]

settings = load_settings()
repository = GoldRepository(settings.gold_dir)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.gold_validation = repository.validate_contract()
    yield


app = FastAPI(
    title=SERVICE_NAME,
    version=API_VERSION,
    description=(
        "Read-only API over the accepted IHMI canonical Gold layer. "
        "The service does not rebuild data, refit models, or expose exact coordinates."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

if settings.cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=False,
        allow_methods=["GET"],
        allow_headers=["*"],
    )


def _error_to_http(exc: Exception) -> HTTPException:
    if isinstance(exc, UnknownDatasetError):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, DatasetContractError):
        return HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, GoldRepositoryError):
        return HTTPException(status_code=503, detail=str(exc))
    return HTTPException(status_code=500, detail="Unexpected API error")


def _dataset_response(
    dataset: str,
    *,
    filters: dict[str, Any] | None = None,
    in_filters: dict[str, list[str]] | None = None,
    sort_by: str | None = None,
    descending: bool = False,
    limit: int,
    offset: int,
) -> DatasetResponse:
    try:
        total, rows = repository.query(
            dataset,
            filters=filters,
            in_filters=in_filters,
            sort_by=sort_by,
            descending=descending,
            limit=limit,
            offset=offset,
        )
    except Exception as exc:
        raise _error_to_http(exc) from exc

    return DatasetResponse(
        dataset=dataset,
        total=total,
        returned=len(rows),
        limit=limit,
        offset=offset,
        rows=rows,
    )


def _page_size(limit: int) -> int:
    return min(max(1, limit), settings.max_page_size)


@app.get("/", tags=["service"])
def root() -> dict[str, Any]:
    return {
        "service": SERVICE_NAME,
        "version": API_VERSION,
        "status_endpoint": "/health",
        "swagger": "/docs",
        "redoc": "/redoc",
        "api_prefix": settings.api_prefix,
        "read_only": True,
    }


@app.get("/health", response_model=HealthResponse, tags=["service"])
def health(request: Request) -> HealthResponse:
    validation = getattr(request.app.state, "gold_validation", None)
    if validation is None:
        validation = repository.validate_contract()
    degraded = bool(
        validation["missing_artifacts"]
        or validation["schema_issues"]
        or validation["forbidden_columns"]
    )
    return HealthResponse(
        status="degraded" if degraded else "ok",
        service=SERVICE_NAME,
        api_version=API_VERSION,
        gold_qa_status=repository.gold_qa_status(),
        expected_marts=len(EXPECTED_MARTS),
        expected_dimensions=len(EXPECTED_DIMENSIONS),
        missing_artifacts=validation["missing_artifacts"],
        schema_issues=validation["schema_issues"],
        forbidden_columns=validation["forbidden_columns"],
        claim_boundaries=CLAIM_BOUNDARIES,
    )


@app.get(f"{settings.api_prefix}/meta", tags=["metadata"])
def metadata() -> dict[str, Any]:
    return {
        "service": SERVICE_NAME,
        "api_version": API_VERSION,
        "architecture": {
            "marts": list(EXPECTED_MARTS),
            "dimensions": list(EXPECTED_DIMENSIONS),
            "mart_count": len(EXPECTED_MARTS),
            "dimension_count": len(EXPECTED_DIMENSIONS),
        },
        "claim_boundaries": CLAIM_BOUNDARIES,
    }


@app.get(
    f"{settings.api_prefix}/datasets",
    response_model=list[DatasetInfo],
    tags=["metadata"],
)
def datasets() -> list[DatasetInfo]:
    try:
        return [DatasetInfo(**row) for row in repository.inventory()]
    except Exception as exc:
        raise _error_to_http(exc) from exc


@app.get(
    f"{settings.api_prefix}/market/monthly",
    response_model=DatasetResponse,
    tags=["market"],
)
def market_monthly(
    city_slug: str | None = None,
    neighborhood_slug: str | None = None,
    analysis_month: str | None = None,
    entity_level: str | None = None,
    market_scope: str | None = None,
    series_kind: str | None = None,
    property_key: str | None = None,
    price_regime_key: str | None = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> DatasetResponse:
    in_filters: dict[str, list[str]] = {}
    if city_slug is not None or neighborhood_slug is not None:
        try:
            keys = repository.resolve_location_keys(
                city_slug=city_slug,
                neighborhood_slug=neighborhood_slug,
            )
        except Exception as exc:
            raise _error_to_http(exc) from exc
        if not keys:
            return DatasetResponse(
                dataset="mart_market_monthly",
                total=0,
                returned=0,
                limit=_page_size(limit),
                offset=offset,
                rows=[],
            )
        in_filters["location_key"] = keys

    return _dataset_response(
        "mart_market_monthly",
        filters={
            "analysis_month": analysis_month,
            "entity_level": entity_level,
            "market_scope": market_scope,
            "series_kind": series_kind,
            "property_key": property_key,
            "price_regime_key": price_regime_key,
        },
        in_filters=in_filters,
        sort_by="analysis_month",
        limit=_page_size(limit),
        offset=offset,
    )


@app.get(
    f"{settings.api_prefix}/market/locations",
    response_model=DatasetResponse,
    tags=["market"],
)
def location_market(
    city_slug: str | None = None,
    neighborhood_slug: str | None = None,
    entity_level: str | None = None,
    property_key: str | None = None,
    price_regime_key: str | None = None,
    reliable_only: bool = False,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> DatasetResponse:
    filters: dict[str, Any] = {
        "city_slug": city_slug,
        "neighborhood_slug": neighborhood_slug,
        "entity_level": entity_level,
        "property_key": property_key,
        "price_regime_key": price_regime_key,
    }
    if reliable_only:
        filters["temperature_reliability_eligible_flag"] = True

    return _dataset_response(
        "mart_location_market",
        filters=filters,
        sort_by="market_temperature_score",
        descending=True,
        limit=_page_size(limit),
        offset=offset,
    )


@app.get(
    f"{settings.api_prefix}/drivers/effects",
    response_model=DatasetResponse,
    tags=["model"],
)
def driver_effects(
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> DatasetResponse:
    return _dataset_response(
        "mart_price_driver_effects",
        limit=_page_size(limit),
        offset=offset,
    )


@app.get(
    f"{settings.api_prefix}/drivers/importance",
    response_model=DatasetResponse,
    tags=["model"],
)
def driver_importance(
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> DatasetResponse:
    return _dataset_response(
        "mart_price_driver_importance",
        sort_by="permutation_importance",
        descending=True,
        limit=_page_size(limit),
        offset=offset,
    )


@app.get(
    f"{settings.api_prefix}/model/quality",
    response_model=DatasetResponse,
    tags=["model"],
)
def model_quality(
    record_type: str | None = None,
    model_name: str | None = None,
    evaluation_split: str | None = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> DatasetResponse:
    return _dataset_response(
        "mart_model_quality",
        filters={
            "record_type": record_type,
            "model_name": model_name,
            "evaluation_split": evaluation_split,
        },
        limit=_page_size(limit),
        offset=offset,
    )


@app.get(
    f"{settings.api_prefix}/seller",
    response_model=DatasetResponse,
    tags=["seller"],
)
def seller_comparison(
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> DatasetResponse:
    return _dataset_response(
        "mart_seller_type",
        limit=_page_size(limit),
        offset=offset,
    )


@app.get(
    f"{settings.api_prefix}/text/signals",
    response_model=DatasetResponse,
    tags=["text"],
)
def text_signals(
    keyword_family: str | None = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> DatasetResponse:
    return _dataset_response(
        "mart_text_signals",
        filters={"keyword_family": keyword_family},
        limit=_page_size(limit),
        offset=offset,
    )


@app.get(
    f"{settings.api_prefix}/text/monthly",
    response_model=DatasetResponse,
    tags=["text"],
)
def text_monthly(
    analysis_month: str | None = None,
    keyword_family: str | None = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> DatasetResponse:
    return _dataset_response(
        "mart_text_monthly",
        filters={
            "analysis_month": analysis_month,
            "keyword_family": keyword_family,
        },
        sort_by="analysis_month",
        limit=_page_size(limit),
        offset=offset,
    )


@app.get(
    f"{settings.api_prefix}/segments",
    response_model=DatasetResponse,
    tags=["segments"],
)
def segment_profiles(
    segment_id: str | None = None,
    property_key: str | None = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> DatasetResponse:
    return _dataset_response(
        "mart_segment_profile",
        filters={
            "segment_id": segment_id,
            "property_key": property_key,
        },
        sort_by="listing_n",
        descending=True,
        limit=_page_size(limit),
        offset=offset,
    )


@app.get(
    f"{settings.api_prefix}/segments/monthly",
    response_model=DatasetResponse,
    tags=["segments"],
)
def segment_monthly(
    analysis_month: str | None = None,
    segment_id: str | None = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> DatasetResponse:
    return _dataset_response(
        "mart_segment_monthly_mix",
        filters={
            "analysis_month": analysis_month,
            "segment_id": segment_id,
        },
        sort_by="analysis_month",
        limit=_page_size(limit),
        offset=offset,
    )


@app.get(
    f"{settings.api_prefix}/dimensions/{{dimension_name}}",
    response_model=DatasetResponse,
    tags=["metadata"],
)
def dimension(
    dimension_name: str,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> DatasetResponse:
    if dimension_name not in EXPECTED_DIMENSIONS:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown dimension. Allowed: {', '.join(EXPECTED_DIMENSIONS)}",
        )
    return _dataset_response(
        dimension_name,
        limit=_page_size(limit),
        offset=offset,
    )
