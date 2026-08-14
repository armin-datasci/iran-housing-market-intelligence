from __future__ import annotations

import argparse
import csv
from pathlib import Path

import polars as pl

from src.common.io_utils import atomic_write_csv
from src.common.paths import DOCS_DIR, OUTPUTS_DIR

CRITICAL_COLUMNS = {
    "source_row_id", "cat2_slug", "cat3_slug", "city_slug", "created_at_month",
    "analysis_month", "price_regime", "type_parse_error_count",
}


def _dictionary_missing_meanings() -> dict[str, str]:
    path = DOCS_DIR / "data_dictionary.csv"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {
            row.get("column_name", ""): row.get("missing_meaning", "")
            for row in csv.DictReader(handle)
            if row.get("column_name")
        }


def _action_for(column: str, missing_rate: float) -> tuple[str, str, str]:
    if column in CRITICAL_COLUMNS:
        return (
            "Investigate and keep null until a documented source rule exists.",
            "This field is required for schema, population, or regime control.",
            "Missing values can remove rows from canonical analytical populations.",
        )
    if column.endswith("_bool"):
        return (
            "Preserve null; do not convert unknown amenities to False.",
            "Unknown and explicit absence are different business states.",
            "Blanket False would bias amenity prevalence and price-driver analysis.",
        )
    if column.endswith(("_toman", "_sqm", "_num", "_years")):
        return (
            "Preserve null in Silver; impute only inside a specific downstream model when justified.",
            "Missing numeric values are not equivalent to zero.",
            "Premature imputation can bias price, area, and model estimates.",
        )
    if missing_rate >= 0.80:
        return (
            "Retain for auditability but treat as low-coverage unless a task explicitly needs it.",
            "Coverage is too low for unrestricted analytical use.",
            "Using the field without a coverage policy may create unstable results.",
        )
    return (
        "Preserve null and apply task-specific applicability rules downstream.",
        "Missingness can mean unavailable, unknown, or not applicable depending on the field.",
        "Global zero/False filling would change the meaning of the source data.",
    )


def run_missingness_audit(input_path: Path, output_path: Path | None = None) -> Path:
    """Profile missingness only; this stage never imputes or mutates Silver values."""
    output_path = output_path or OUTPUTS_DIR / "tables" / "milestone_2" / "missingness" / "missingness_action_table.csv"
    scan = pl.scan_parquet(input_path)
    schema = scan.collect_schema()
    columns = schema.names()
    expressions = [pl.len().alias("__rows")]
    for column in columns:
        dtype = schema[column]
        if dtype == pl.String:
            value = pl.col(column).str.strip_chars()
            missing = value.is_null() | value.str.to_lowercase().is_in(["", "null", "<null>", "none", "nan", "n/a", "na"])
        else:
            missing = pl.col(column).is_null()
        expressions.append(missing.sum().alias(column))
    values = scan.select(expressions).collect(engine="streaming").row(0, named=True)
    n = int(values.pop("__rows"))
    meanings = _dictionary_missing_meanings()
    rows = []
    for column in columns:
        missing = int(values[column] or 0)
        rate = missing / n if n else 0.0
        action, reason, effect = _action_for(column, rate)
        rows.append({
            "column": column,
            "issue": "missing_value",
            "affected_count": missing,
            "affected_rate": rate,
            "missing_error_meaning": meanings.get(column, "Unknown / unavailable / not applicable depending on the field contract."),
            "action": action,
            "reason": reason,
            "possible_effect": effect,
        })
    atomic_write_csv(pl.DataFrame(rows), output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the canonical M2 missingness action table.")
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    print(run_missingness_audit(args.input))


if __name__ == "__main__":
    main()
