from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class DatasetResponse(BaseModel):
    dataset: str
    total: int = Field(ge=0)
    returned: int = Field(ge=0)
    limit: int = Field(ge=1)
    offset: int = Field(ge=0)
    rows: list[dict[str, Any]]


class DatasetInfo(BaseModel):
    name: str
    kind: Literal["mart", "dimension"]
    rows: int = Field(ge=0)
    columns: list[str]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    service: str
    api_version: str
    gold_qa_status: str
    expected_marts: int
    expected_dimensions: int
    missing_artifacts: list[str]
    schema_issues: dict[str, list[str]]
    forbidden_columns: dict[str, list[str]]
    claim_boundaries: list[str]
