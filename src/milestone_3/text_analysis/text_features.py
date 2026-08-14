from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl

from src.common.config import setting
from src.common.io_utils import atomic_sink_parquet, atomic_write_csv, atomic_write_json
from src.common.paths import OUTPUTS_DIR, relative_to_project
from src.common.validation import Check, checks_frame, make_check, summarize_checks
from src.milestone_3.text_analysis.text_rules import KEYWORD_RULES, KEYWORD_RULE_VERSION

VERSION = "m3-text-features-v1"
SALES_BASE = OUTPUTS_DIR / "model_artifacts" / "milestone_3" / "analysis_populations" / "sales_analysis_base.parquet"
TEXT_PARQUET = OUTPUTS_DIR / "model_artifacts" / "milestone_3" / "text_analysis" / "text_features.parquet"
PROGRESS_WIDTH = 30


def show_progress(percent: int, label: str, *, final: bool = False) -> None:
    pct = max(0, min(100, int(percent)))
    filled = int(PROGRESS_WIDTH * pct / 100)
    bar = "#" * filled + "-" * (PROGRESS_WIDTH - filled)
    print(
        f"\rM3 text features [{bar}] {pct:3d}% complete | {100-pct:3d}% remaining | {label[:38]:38s}",
        end="\n" if final else "", flush=True,
    )


def _combined_text() -> pl.Expr:
    return pl.concat_str(
        [pl.col("title_normalized").fill_null(""), pl.col("description_normalized").fill_null("")],
        separator=" ",
    ).str.replace_all(r"\s+", " ").str.strip_chars().alias("_combined_text")


def _flag_name(key: str) -> str:
    return f"keyword_{key}_flag"


def _mask_excerpt(value: Any, maximum: int = 220) -> str:
    text = "" if value is None else str(value)
    text = re.sub(r"[0-9۰-۹٠-٩]{4,}", "[number]", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:maximum]


def _precision_summary(review_path: Path, minimum_reviewed: int, threshold: float) -> pl.DataFrame:
    review = pd.read_csv(review_path)
    if "manual_relevance_label" not in review:
        review["manual_relevance_label"] = pd.NA
    review["manual_relevance_label"] = pd.to_numeric(review["manual_relevance_label"], errors="coerce")
    rows: list[dict[str, Any]] = []
    for key, spec in KEYWORD_RULES.items():
        part = review.loc[review["keyword"] == key]
        labels = part["manual_relevance_label"].dropna()
        reviewed_n = int(len(labels))
        relevant_n = int((labels == 1).sum())
        precision = float(relevant_n / reviewed_n) if reviewed_n else None
        passed = reviewed_n >= minimum_reviewed and precision is not None and precision >= threshold
        rows.append({
            "keyword": key,
            "keyword_fa": spec["keyword_fa"],
            "reviewed_n": reviewed_n,
            "relevant_n": relevant_n,
            "manual_precision": precision,
            "minimum_reviewed_required": minimum_reviewed,
            "precision_threshold": threshold,
            "validation_status": "PASS" if passed else "REVIEW",
            "include_in_controlled_analysis": passed,
            "rule_version": KEYWORD_RULE_VERSION,
        })
    return pl.from_pandas(pd.DataFrame(rows))


def run(sales_base: Path = SALES_BASE) -> dict[str, Path]:
    sales_base = sales_base.resolve()
    if not sales_base.exists():
        raise FileNotFoundError(f"Sales analysis base not found: {sales_base}")
    sample_per_keyword = int(setting("milestone_3", "text", "validation_sample_per_keyword", default=40))
    minimum_reviewed = int(setting("milestone_3", "text", "minimum_reviewed_per_keyword", default=20))
    threshold = float(setting("milestone_3", "text", "precision_threshold", default=0.80))
    seed = int(setting("project", "random_seed", default=42))

    table_dir = OUTPUTS_DIR / "tables" / "milestone_3" / "text_analysis"
    qa_dir = OUTPUTS_DIR / "qa" / "milestone_3" / "text_analysis"
    TEXT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    qa_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    show_progress(0, "validating text source")

    scan = pl.scan_parquet(sales_base)
    columns = set(scan.collect_schema().names())
    required = {"source_row_id", "analysis_month", "city_slug", "neighborhood_slug", "title_normalized", "description_normalized"}
    missing = sorted(required - columns)
    if missing:
        raise ValueError(f"Sales base missing text columns: {missing}")

    work = scan.with_columns(_combined_text())
    expressions: list[pl.Expr] = []
    for key, spec in KEYWORD_RULES.items():
        expressions.append(pl.col("_combined_text").str.contains(spec["pattern"]).fill_null(False).alias(_flag_name(key)))
    keep = [column for column in ["source_row_id", "analysis_month", "city_slug", "neighborhood_slug", "cat3_slug", "property_family"] if column in columns]
    features = work.with_columns(expressions).select([*keep, *[_flag_name(key) for key in KEYWORD_RULES]])
    atomic_sink_parquet(features, TEXT_PARQUET)
    show_progress(45, "keyword flags written")

    flag_cols = [_flag_name(key) for key in KEYWORD_RULES]
    freq_exprs: list[pl.Expr] = [pl.len().alias("population_n")]
    for flag in flag_cols:
        freq_exprs.append(pl.col(flag).cast(pl.Int64).sum().alias(flag))
    totals = pl.scan_parquet(TEXT_PARQUET).select(freq_exprs).collect(engine="streaming").row(0, named=True)
    frequency = pl.DataFrame([
        {
            "keyword": key,
            "keyword_fa": spec["keyword_fa"],
            "positive_n": int(totals[_flag_name(key)] or 0),
            "population_n": int(totals["population_n"] or 0),
            "positive_rate": int(totals[_flag_name(key)] or 0) / int(totals["population_n"]) if int(totals["population_n"] or 0) else None,
            "rule_version": KEYWORD_RULE_VERSION,
        }
        for key, spec in KEYWORD_RULES.items()
    ])
    frequency_path = table_dir / "text_keyword_frequency.csv"
    atomic_write_csv(frequency, frequency_path)
    show_progress(62, "frequency summary complete")

    review_path = qa_dir / "keyword_manual_validation.csv"
    if not review_path.exists():
        source = pl.read_parquet(sales_base, columns=["source_row_id", "title_normalized", "description_normalized"])
        flags = pl.read_parquet(TEXT_PARQUET)
        merged = flags.join(source, on="source_row_id", how="left").to_pandas()
        review_rows: list[pd.DataFrame] = []
        for index, (key, spec) in enumerate(KEYWORD_RULES.items()):
            part = merged.loc[merged[_flag_name(key)].fillna(False).astype(bool)].copy()
            if len(part) > sample_per_keyword:
                part = part.sample(n=sample_per_keyword, random_state=seed + index)
            part["keyword"] = key
            part["keyword_fa"] = spec["keyword_fa"]
            part["text_excerpt"] = (part["title_normalized"].fillna("") + " | " + part["description_normalized"].fillna("")).map(_mask_excerpt)
            part["manual_relevance_label"] = pd.NA
            part["manual_review_note"] = ""
            review_rows.append(part[["keyword", "keyword_fa", "source_row_id", "text_excerpt", "manual_relevance_label", "manual_review_note"]])
        review = pd.concat(review_rows, ignore_index=True) if review_rows else pd.DataFrame(columns=["keyword", "keyword_fa", "source_row_id", "text_excerpt", "manual_relevance_label", "manual_review_note"])
        atomic_write_csv(pl.from_pandas(review), review_path)
    show_progress(78, "manual validation sample ready")

    precision = _precision_summary(review_path, minimum_reviewed, threshold)
    precision_path = qa_dir / "keyword_precision_summary.csv"
    atomic_write_csv(precision, precision_path)
    unvalidated = precision.filter(pl.col("validation_status") != "PASS").height
    checks: list[Check] = [
        make_check("text_feature_population_nonempty", "text", int(totals["population_n"] or 0), ">0", int(totals["population_n"] or 0) > 0),
        make_check("keyword_families_present", "text", len(KEYWORD_RULES), ">=6", len(KEYWORD_RULES) >= 6),
        make_check(
            "manual_precision_validation", "text", len(KEYWORD_RULES) - unvalidated, len(KEYWORD_RULES), unvalidated == 0,
            critical=False, review_on_fail=True,
            notes="Fill manual_relevance_label with 1=relevant or 0=false positive. Controlled text-price analysis only uses PASS families.",
        ),
    ]
    checks_path = qa_dir / "text_feature_checks.csv"
    manifest_path = qa_dir / "text_feature_manifest.json"
    atomic_write_csv(checks_frame(checks), checks_path)
    status = summarize_checks(checks)
    atomic_write_json(
        {
            "version": VERSION,
            "status": status,
            "input": relative_to_project(sales_base),
            "output": relative_to_project(TEXT_PARQUET),
            "keyword_rule_version": KEYWORD_RULE_VERSION,
            "manual_validation_file": relative_to_project(review_path),
            "validation_instruction": "Set manual_relevance_label to 1 or 0 for sampled positive matches; rerun this module to refresh precision summary.",
            "outputs": {"frequency": relative_to_project(frequency_path), "precision": relative_to_project(precision_path), "checks": relative_to_project(checks_path)},
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }, manifest_path,
    )
    show_progress(100, f"complete; {unvalidated} keyword families need review", final=True)
    return {"features": TEXT_PARQUET, "frequency": frequency_path, "manual_validation": review_path, "precision": precision_path, "checks": checks_path, "manifest": manifest_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract validated-rule Persian text signals and prepare manual precision review.")
    parser.add_argument("--sales-base", type=Path, default=SALES_BASE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run(args.sales_base)
    print("M3 TEXT FEATURES COMPLETED")
    for name, path in outputs.items():
        print(f"{name}: {relative_to_project(path)}")


if __name__ == "__main__":
    main()
