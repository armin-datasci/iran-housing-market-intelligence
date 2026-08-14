from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import polars as pl

from src.milestone_4.gold.contracts import (
    EXPECTED_DIMENSIONS,
    EXPECTED_MARTS,
    FORBIDDEN_COLUMN_TOKENS,
    REQUIRED_ARTIFACT_COLUMNS,
)


class GoldRepositoryError(RuntimeError):
    pass


class UnknownDatasetError(GoldRepositoryError):
    pass


class DatasetContractError(GoldRepositoryError):
    pass


class GoldRepository:
    """Read-only access to the canonical IHMI Gold layer.

    The repository never writes Gold and never refits analytical models. API
    responses are filters/projections of accepted Gold Parquet artifacts only.
    """

    def __init__(self, gold_dir: Path):
        self.gold_dir = Path(gold_dir).resolve()
        self.marts_dir = self.gold_dir / "marts"
        self.dimensions_dir = self.gold_dir / "dimensions"
        self.qa_dir = self.gold_dir / "qa"

    @property
    def allowed_datasets(self) -> tuple[str, ...]:
        return tuple(EXPECTED_MARTS) + tuple(EXPECTED_DIMENSIONS)

    def dataset_kind(self, name: str) -> str:
        if name in EXPECTED_MARTS:
            return "mart"
        if name in EXPECTED_DIMENSIONS:
            return "dimension"
        raise UnknownDatasetError(
            f"Dataset is not part of the canonical Gold contract: {name}"
        )

    def path_for(self, name: str) -> Path:
        kind = self.dataset_kind(name)
        base = self.marts_dir if kind == "mart" else self.dimensions_dir
        return base / f"{name}.parquet"

    def _scan(self, name: str) -> pl.LazyFrame:
        path = self.path_for(name)
        if not path.exists():
            raise GoldRepositoryError(f"Required Gold artifact is missing: {path}")
        return pl.scan_parquet(path)

    def columns(self, name: str) -> list[str]:
        return self._scan(name).collect_schema().names()

    def row_count(self, name: str) -> int:
        return int(self._scan(name).select(pl.len().alias("n")).collect().item())

    def validate_contract(self) -> dict[str, Any]:
        missing_artifacts: list[str] = []
        schema_issues: dict[str, list[str]] = {}
        forbidden_columns: dict[str, list[str]] = {}

        for name in self.allowed_datasets:
            path = self.path_for(name)
            if not path.exists():
                missing_artifacts.append(name)
                continue

            columns = self.columns(name)
            required = set(REQUIRED_ARTIFACT_COLUMNS[name])
            missing_columns = sorted(required - set(columns))
            if missing_columns:
                schema_issues[name] = missing_columns

            forbidden = sorted(
                col
                for col in columns
                if any(token in col.lower() for token in FORBIDDEN_COLUMN_TOKENS)
            )
            if forbidden:
                forbidden_columns[name] = forbidden

        return {
            "missing_artifacts": sorted(missing_artifacts),
            "schema_issues": schema_issues,
            "forbidden_columns": forbidden_columns,
        }

    def gold_qa_status(self) -> str:
        candidates = (
            self.qa_dir / "gold_qa_manifest.json",
            self.qa_dir / "gold_manifest.json",
        )
        for path in candidates:
            if not path.exists():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8-sig"))
            except (OSError, json.JSONDecodeError):
                continue
            direct = payload.get("overall_status")
            if direct:
                return str(direct).upper()
            nested = payload.get("status")
            if isinstance(nested, dict) and nested.get("overall_status"):
                return str(nested["overall_status"]).upper()
        return "UNKNOWN"

    def inventory(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for name in self.allowed_datasets:
            path = self.path_for(name)
            if not path.exists():
                continue
            rows.append(
                {
                    "name": name,
                    "kind": self.dataset_kind(name),
                    "rows": self.row_count(name),
                    "columns": self.columns(name),
                }
            )
        return rows

    def query(
        self,
        name: str,
        *,
        filters: dict[str, Any] | None = None,
        in_filters: dict[str, Iterable[Any]] | None = None,
        sort_by: str | None = None,
        descending: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[int, list[dict[str, Any]]]:
        frame = self._scan(name)
        columns = set(frame.collect_schema().names())

        for column, value in (filters or {}).items():
            if value is None:
                continue
            if column not in columns:
                raise DatasetContractError(
                    f"{name} does not contain filter column {column!r}"
                )
            if isinstance(value, bool):
                frame = frame.filter(
                    pl.col(column).cast(pl.Boolean, strict=False) == value
                )
            elif isinstance(value, (int, float)):
                frame = frame.filter(
                    pl.col(column).cast(pl.Float64, strict=False) == float(value)
                )
            else:
                frame = frame.filter(
                    pl.col(column).cast(pl.String, strict=False) == str(value)
                )

        for column, values in (in_filters or {}).items():
            values = list(values)
            if not values:
                return 0, []
            if column not in columns:
                raise DatasetContractError(
                    f"{name} does not contain filter column {column!r}"
                )
            frame = frame.filter(
                pl.col(column)
                .cast(pl.String, strict=False)
                .is_in([str(x) for x in values])
            )

        total = int(frame.select(pl.len().alias("n")).collect().item())

        if sort_by:
            if sort_by not in columns:
                raise DatasetContractError(
                    f"{name} does not contain sort column {sort_by!r}"
                )
            frame = frame.sort(sort_by, descending=descending, nulls_last=True)

        result = frame.slice(offset, limit).collect(engine="streaming")
        return total, result.to_dicts()

    def resolve_location_keys(
        self,
        *,
        city_slug: str | None = None,
        neighborhood_slug: str | None = None,
    ) -> list[str]:
        if city_slug is None and neighborhood_slug is None:
            return []

        frame = self._scan("dim_location")
        columns = set(frame.collect_schema().names())

        if city_slug is not None:
            if "city_slug" not in columns:
                raise DatasetContractError("dim_location does not contain city_slug")
            frame = frame.filter(
                pl.col("city_slug").cast(pl.String, strict=False) == city_slug
            )

        if neighborhood_slug is not None:
            if "neighborhood_slug" not in columns:
                raise DatasetContractError(
                    "dim_location does not contain neighborhood_slug"
                )
            frame = frame.filter(
                pl.col("neighborhood_slug").cast(pl.String, strict=False)
                == neighborhood_slug
            )

        return (
            frame.select(pl.col("location_key").cast(pl.String, strict=False))
            .filter(pl.col("location_key").is_not_null())
            .unique()
            .collect(engine="streaming")
            .get_column("location_key")
            .to_list()
        )
