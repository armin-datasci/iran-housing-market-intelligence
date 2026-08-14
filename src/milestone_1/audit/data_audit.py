from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

from src.common.config import configured_path
from src.common.io_utils import atomic_write_csv, atomic_write_json
from src.common.paths import OUTPUTS_DIR, relative_to_project

try:
    import psutil
except ImportError:
    psutil = None

AUDIT_VERSION = "m1-raw-audit-v1.2"
PROGRESS_WIDTH = 30
NULL_LIKE = ["", "null", "<null>", "none", "nan", "n/a", "na"]
SPECIAL_LABELS = [
    "\u0646\u0627\u0645\u0634\u062e\u0635",
    "\u0633\u0627\u06cc\u0631",
    "\u062a\u0648\u0627\u0641\u0642\u06cc",
    "\u062c\u0647\u062a \u0645\u0639\u0627\u0648\u0636\u0647",
    "\u0645\u062c\u0627\u0646\u06cc",
    "\u0645\u0642\u0637\u0648\u0639",
    "\u0646\u062f\u0627\u0631\u062f",
    "\u0647\u06cc\u0686",
    "unselect",
    "\u0628\u062f\u0648\u0646 \u0627\u062a\u0627\u0642",
    "\u0642\u0628\u0644 \u0627\u0632 \u06f1\u06f3\u06f7\u06f0",
    "30+",
]
NUMERIC_COLUMNS = [
    "building_size", "land_size", "construction_year", "price_value",
    "rent_value", "credit_value", "transformed_credit", "transformed_rent",
    "cost_per_extra_person", "rent_price_on_regular_days",
    "rent_price_on_special_days", "rent_price_at_weekends",
    "regular_person_capacity", "extra_person_capacity", "floor",
    "total_floors_count", "unit_per_floor", "rooms_count",
    "location_latitude", "location_longitude", "location_radius",
]
NONNEGATIVE_COLUMNS = [
    "building_size", "land_size", "price_value", "rent_value", "credit_value",
    "rooms_count", "floor", "total_floors_count", "unit_per_floor",
    "regular_person_capacity", "extra_person_capacity", "location_radius",
]
DISTRIBUTIONS = {
    "property_type": "cat3_slug",
    "city": "city_slug",
    "month": "created_at_month",
    "user_type": "user_type",
}
PROBABLE_DUPLICATE_COLUMNS = [
    "city_slug", "neighborhood_slug", "cat3_slug", "price_value",
    "building_size", "rooms_count", "created_at_month", "user_type",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def rss_mib() -> float | None:
    if psutil is None:
        return None
    return round(psutil.Process().memory_info().rss / (1024**2), 2)


def show_progress(percent: int, label: str, *, final: bool = False) -> None:
    """Display approximate stage progress without another dependency."""
    pct = max(0, min(100, int(percent)))
    filled = int(PROGRESS_WIDTH * pct / 100)
    bar = "#" * filled + "-" * (PROGRESS_WIDTH - filled)
    print(
        f"\rAudit progress [{bar}] {pct:3d}% complete | "
        f"{100 - pct:3d}% remaining | {label[:44]:44s}",
        end="\n" if final else "",
        flush=True,
    )


def _stage(log: list[dict[str, Any]], name: str, started: float) -> float:
    now = time.perf_counter()
    log.append(
        {
            "stage": name,
            "elapsed_seconds": round(now - started, 4),
            "rss_mib_after_stage": rss_mib(),
        }
    )
    return now


def _text(column: str) -> pl.Expr:
    return pl.col(column).cast(pl.String).str.strip_chars()


def _missing(column: str) -> pl.Expr:
    text = _text(column)
    return text.is_null() | text.str.to_lowercase().is_in(NULL_LIKE)


def _numeric(column: str) -> pl.Expr:
    return (
        _text(column)
        .str.replace_all(",", "")
        .str.replace_all("\u066c", "")
        .str.replace_all(" ", "")
        .cast(pl.Float64, strict=False)
    )


def _scan(path: Path, columns: list[str], max_rows: int | None) -> pl.LazyFrame:
    frame = pl.scan_csv(
        path,
        has_header=True,
        schema_overrides={column: pl.String for column in columns},
        infer_schema=False,
        try_parse_dates=False,
        ignore_errors=False,
        truncate_ragged_lines=False,
        low_memory=True,
        rechunk=False,
    )
    return frame.head(max_rows) if max_rows is not None else frame


def _report_row(
    section: str,
    metric: str,
    value: Any,
    *,
    rate: float | None = None,
    column: str | None = None,
    scope: str = "raw",
    status: str = "PASS",
    notes: str = "",
) -> dict[str, Any]:
    return {
        "section": section,
        "scope": scope,
        "column_name": column,
        "metric": metric,
        "value": str(value),
        "rate": rate,
        "status": status,
        "notes": notes,
    }


def column_profile(scan: pl.LazyFrame, columns: list[str]) -> tuple[pl.DataFrame, int]:
    exprs: list[pl.Expr] = [pl.len().alias("__rows")]
    for column in columns:
        missing = _missing(column)
        value = pl.when(missing).then(None).otherwise(_text(column))
        exprs += [
            missing.sum().alias(f"{column}__missing"),
            value.drop_nulls().n_unique().alias(f"{column}__unique"),
        ]

    values = scan.select(exprs).collect(engine="streaming").row(0, named=True)
    n = int(values["__rows"])
    rows = []
    for column in columns:
        missing = int(values[f"{column}__missing"] or 0)
        rows.append(
            {
                "column_name": column,
                "raw_dtype": "String",
                "total_rows": n,
                "missing_count": missing,
                "missing_rate": missing / n if n else None,
                "unique_count": int(values[f"{column}__unique"] or 0),
            }
        )
    return pl.DataFrame(rows), n


def parse_diagnostics(
    scan: pl.LazyFrame,
    columns: list[str],
    row_count: int,
) -> pl.DataFrame:
    active = [column for column in NUMERIC_COLUMNS if column in columns]
    if not active:
        return pl.DataFrame()

    exprs: list[pl.Expr] = []
    for column in active:
        text = _text(column)
        missing = _missing(column)
        special = text.is_in(SPECIAL_LABELS).fill_null(False)
        number = _numeric(column)
        candidate = (~missing) & (~special)
        finite = number.is_not_null() & number.is_finite().fill_null(False)
        valid = candidate & finite
        failed = candidate & (~finite)
        exprs += [
            missing.sum().alias(f"{column}__missing"),
            special.sum().alias(f"{column}__semantic"),
            valid.sum().alias(f"{column}__valid"),
            failed.sum().alias(f"{column}__failed"),
            number.filter(valid).min().alias(f"{column}__min"),
            number.filter(valid).max().alias(f"{column}__max"),
        ]

    values = scan.select(exprs).collect(engine="streaming").row(0, named=True)
    rows = []
    for column in active:
        valid = int(values[f"{column}__valid"] or 0)
        failed = int(values[f"{column}__failed"] or 0)
        candidate = valid + failed
        rows.append(
            {
                "column_name": column,
                "total_rows": row_count,
                "missing_count": int(values[f"{column}__missing"] or 0),
                "semantic_label_count": int(values[f"{column}__semantic"] or 0),
                "valid_numeric_count": valid,
                "parse_failure_count": failed,
                "parse_failure_rate": failed / candidate if candidate else None,
                "minimum_numeric": values[f"{column}__min"],
                "maximum_numeric": values[f"{column}__max"],
            }
        )
    return pl.DataFrame(rows)


def range_dependency_checks(scan: pl.LazyFrame, columns: list[str]) -> list[dict[str, Any]]:
    available = set(columns)
    exprs: list[pl.Expr] = []
    definitions: list[tuple[str, str, str]] = []

    for column in NONNEGATIVE_COLUMNS:
        if column in available:
            exprs.append((_numeric(column) < 0).fill_null(False).sum().alias(f"neg__{column}"))
            definitions.append((f"neg__{column}", "range_check", column))

    if {"location_latitude", "location_longitude"}.issubset(available):
        lat, lon = _numeric("location_latitude"), _numeric("location_longitude")
        exprs += [
            (_missing("location_latitude") ^ _missing("location_longitude"))
            .fill_null(False).sum().alias("coordinate_pair_incomplete"),
            ((lat < -90) | (lat > 90)).fill_null(False).sum().alias("latitude_outside_world_range"),
            ((lon < -180) | (lon > 180)).fill_null(False).sum().alias("longitude_outside_world_range"),
            (lat.is_not_null() & lon.is_not_null() & ~lat.is_between(24.0, 40.5, closed="both"))
            .fill_null(False).sum().alias("latitude_outside_iran_window"),
            (lat.is_not_null() & lon.is_not_null() & ~lon.is_between(44.0, 64.0, closed="both"))
            .fill_null(False).sum().alias("longitude_outside_iran_window"),
        ]
        definitions += [
            ("coordinate_pair_incomplete", "dependency_check", "location_latitude/location_longitude"),
            ("latitude_outside_world_range", "coordinate_check", "location_latitude"),
            ("longitude_outside_world_range", "coordinate_check", "location_longitude"),
            ("latitude_outside_iran_window", "coordinate_check", "location_latitude"),
            ("longitude_outside_iran_window", "coordinate_check", "location_longitude"),
        ]

    if {"floor", "total_floors_count"}.issubset(available):
        floor, total = _numeric("floor"), _numeric("total_floors_count")
        exprs.append(
            (floor.is_not_null() & total.is_not_null() & (floor > total))
            .fill_null(False).sum().alias("floor_above_total_floors")
        )
        definitions.append(("floor_above_total_floors", "dependency_check", "floor/total_floors_count"))

    if not exprs:
        return []
    values = scan.select(exprs).collect(engine="streaming").row(0, named=True)
    rows = []
    for metric, section, column in definitions:
        count = int(values[metric] or 0)
        rows.append(
            _report_row(
                section,
                metric,
                count,
                column=column,
                status="REVIEW" if count else "PASS",
                notes="Raw diagnostic only; M2 owns canonical cleaning and eligibility.",
            )
        )
    return rows


def duplicate_diagnostics(
    scan: pl.LazyFrame,
    columns: list[str],
) -> tuple[dict[str, int], dict[str, Any]]:
    probable = [column for column in PROBABLE_DUPLICATE_COLUMNS if column in columns]
    required = [column for column in ["city_slug", "cat3_slug", "price_value", "building_size"] if column in columns]
    probable_enabled = len(required) == 4 and len(probable) >= 6

    exprs = [
        pl.struct(columns).hash(seed=17).alias("_exact_h1"),
        pl.struct(columns).hash(seed=97).alias("_exact_h2"),
    ]
    if probable_enabled:
        exprs += [
            pl.all_horizontal(*[~_missing(column) for column in required]).alias("_probable_ok"),
            pl.struct(probable).hash(seed=31).alias("_probable_h1"),
            pl.struct(probable).hash(seed=73).alias("_probable_h2"),
        ]

    hashes = scan.select(exprs).collect(engine="streaming")
    exact_groups = hashes.group_by(["_exact_h1", "_exact_h2"]).len().filter(pl.col("len") > 1)
    exact = {
        "duplicate_group_count": exact_groups.height,
        "rows_in_duplicate_groups": int(exact_groups.get_column("len").sum() or 0),
        "excess_duplicate_rows": int((exact_groups.get_column("len") - 1).sum() or 0),
    }

    probable_result: dict[str, Any] = {
        "candidate_group_count": 0,
        "candidate_rows": 0,
        "key_columns": probable,
    }
    if probable_enabled:
        groups = (
            hashes.filter(pl.col("_probable_ok"))
            .group_by(["_probable_h1", "_probable_h2"])
            .len()
            .filter(pl.col("len") > 1)
        )
        probable_result["candidate_group_count"] = groups.height
        probable_result["candidate_rows"] = int(groups.get_column("len").sum() or 0)
    return exact, probable_result


def required_distributions(scan: pl.LazyFrame, columns: list[str]) -> pl.DataFrame:
    active = {label: column for label, column in DISTRIBUTIONS.items() if column in set(columns)}
    if not active:
        return pl.DataFrame()
    values = scan.select([_text(column).alias(label) for label, column in active.items()]).collect(engine="streaming")
    rows = []
    for label, column in active.items():
        counts = values.select(pl.col(label).alias("value")).drop_nulls().group_by("value").len().sort("len", descending=True)
        for rank, item in enumerate(counts.iter_rows(named=True), start=1):
            rows.append(
                {
                    "dimension": label,
                    "source_column": column,
                    "rank": rank,
                    "value": item["value"],
                    "row_count": int(item["len"]),
                }
            )
    return pl.DataFrame(rows)


def main_report(
    profile: pl.DataFrame,
    parse: pl.DataFrame,
    checks: list[dict[str, Any]],
    distributions: pl.DataFrame,
    exact: dict[str, int],
    probable: dict[str, Any],
    row_count: int,
    column_count: int,
    stage_log: list[dict[str, Any]],
) -> pl.DataFrame:
    rows = [
        _report_row("dataset_summary", "row_count", row_count),
        _report_row("dataset_summary", "column_count", column_count),
        _report_row(
            "duplicate_diagnostic",
            "exact_excess_duplicate_rows",
            exact["excess_duplicate_rows"],
            rate=exact["excess_duplicate_rows"] / row_count if row_count else None,
            status="REVIEW" if exact["excess_duplicate_rows"] else "PASS",
            notes="M1 diagnostic only; M2 owns final duplicate flags and Entity Resolution.",
        ),
        _report_row(
            "duplicate_diagnostic",
            "probable_duplicate_candidate_rows",
            probable["candidate_rows"],
            rate=probable["candidate_rows"] / row_count if row_count else None,
            status="REVIEW" if probable["candidate_rows"] else "PASS",
            notes=f"Conservative composite-key diagnostic using {probable['key_columns']}; M2 owns canonical ER.",
        ),
    ]

    for item in profile.iter_rows(named=True):
        rows += [
            _report_row(
                "column_profile", "missing_count", item["missing_count"],
                column=item["column_name"], rate=item["missing_rate"],
                status="REVIEW" if (item["missing_rate"] or 0) >= 0.80 else "PASS",
                notes="Missing is reported, not replaced with zero.",
            ),
            _report_row(
                "column_profile", "unique_count", item["unique_count"],
                column=item["column_name"], notes="Exact non-missing distinct count from Raw.",
            ),
        ]

    if not parse.is_empty():
        for item in parse.iter_rows(named=True):
            rows.append(
                _report_row(
                    "parse_quality", "text_to_number_parse_failure_count",
                    item["parse_failure_count"], column=item["column_name"],
                    rate=item["parse_failure_rate"],
                    status="REVIEW" if item["parse_failure_count"] else "PASS",
                    notes="Discovery only; canonical parsing and Persian normalization are M2 responsibilities.",
                )
            )
    rows.extend(checks)

    if not distributions.is_empty():
        for item in distributions.iter_rows(named=True):
            rows.append(
                _report_row(
                    "distribution", "row_count", item["row_count"],
                    column=item["source_column"], scope=f"raw:{item['dimension']}",
                    rate=item["row_count"] / row_count if row_count else None,
                    notes=f"value={item['value']}; rank={item['rank']}",
                )
            )

    memory = [item["rss_mib_after_stage"] for item in stage_log if item["rss_mib_after_stage"] is not None]
    rows.append(
        _report_row(
            "performance", "observed_process_rss_mib_max",
            max(memory) if memory else "not_available", scope="execution",
            status="PASS" if memory else "REVIEW",
            notes="Stage-level process memory observation via psutil.",
        )
    )
    return pl.DataFrame(rows)


def run_audit(
    raw_path: Path | None = None,
    *,
    main_output: Path | None = None,
    qa_dir: Path | None = None,
    max_rows: int | None = None,
) -> dict[str, Path]:
    """Run the required M1 Raw audit with vectorized Polars operations."""
    raw = (raw_path or configured_path("raw_source")).resolve()
    if not raw.exists():
        raise FileNotFoundError(f"Raw source not found: {raw}")

    main_output = main_output or OUTPUTS_DIR / "tables" / "milestone_1" / "data_audit_report.csv"
    qa_dir = qa_dir or OUTPUTS_DIR / "qa" / "milestone_1"
    qa_dir.mkdir(parents=True, exist_ok=True)

    started, stage_started = time.perf_counter(), time.perf_counter()
    started_at = utc_now()
    stages: list[dict[str, Any]] = []

    show_progress(0, "starting audit")
    columns = pl.scan_csv(raw, has_header=True, infer_schema=False).collect_schema().names()
    scan = _scan(raw, columns, max_rows)
    stage_started = _stage(stages, "schema", stage_started)
    show_progress(8, "Raw schema loaded")

    profile, row_count = column_profile(scan, columns)
    stage_started = _stage(stages, "column_profile", stage_started)
    show_progress(38, "missing and unique profile complete")

    parse = parse_diagnostics(scan, columns, row_count)
    checks = range_dependency_checks(scan, columns)
    stage_started = _stage(stages, "parse_range_dependency", stage_started)
    show_progress(62, "parse, range and dependency checks complete")

    exact, probable = duplicate_diagnostics(scan, columns)
    stage_started = _stage(stages, "duplicate_diagnostics", stage_started)
    show_progress(82, "duplicate diagnostics complete")

    distributions = required_distributions(scan, columns)
    stage_started = _stage(stages, "required_distributions", stage_started)
    show_progress(94, "required distributions complete")

    report = main_report(
        profile, parse, checks, distributions, exact, probable,
        row_count, len(columns), stages,
    )
    outputs = {
        "main_report": main_output,
        "profile": qa_dir / "raw_column_profile.csv",
        "parse": qa_dir / "raw_parse_diagnostics.csv",
        "performance": qa_dir / "performance_log.csv",
        "manifest": qa_dir / "data_audit_manifest.json",
    }
    atomic_write_csv(report, outputs["main_report"])
    atomic_write_csv(profile, outputs["profile"])
    atomic_write_csv(parse, outputs["parse"])
    atomic_write_csv(pl.DataFrame(stages), outputs["performance"])
    atomic_write_json(
        {
            "audit_version": AUDIT_VERSION,
            "started_at_utc": started_at,
            "finished_at_utc": utc_now(),
            "elapsed_seconds": round(time.perf_counter() - started, 4),
            "input": {
                "path": relative_to_project(raw),
                "size_bytes": raw.stat().st_size,
                "row_count_scanned": row_count,
                "column_count": len(columns),
                "development_row_limit": max_rows,
            },
            "design": {
                "engine": "polars",
                "vectorized": True,
                "row_wise_python_scan": False,
                "raw_mutated": False,
                "silver_dependency": False,
                "probable_duplicate_is_diagnostic_only": True,
            },
            "outputs": {name: relative_to_project(path) for name, path in outputs.items()},
        },
        outputs["manifest"],
    )
    show_progress(100, "audit complete", final=True)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Milestone 1 Raw-first vectorized data audit.")
    parser.add_argument("--raw", type=Path, default=None)
    parser.add_argument(
        "--max-rows", type=int, default=None,
        help="Development smoke test only; final notebook must scan the full Raw dataset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_audit(args.raw, max_rows=args.max_rows)
    print("M1 DATA AUDIT COMPLETED")
    for name, path in outputs.items():
        print(f"{name}: {relative_to_project(path)}")


if __name__ == "__main__":
    main()