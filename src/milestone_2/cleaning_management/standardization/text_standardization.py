"""Lightweight Persian/Arabic character normalization for title/description."""

from __future__ import annotations

import polars as pl


def normalize_basic_text_expr(column_name: str) -> pl.Expr:
    expr = pl.col(column_name).cast(pl.String)
    for old, new in [
        ("ي", "ی"), ("ك", "ک"), ("\u00A0", " "), ("\u2007", " "), ("\u202F", " "),
    ]:
        expr = expr.str.replace_all(old, new, literal=True)
    for old, new in zip("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789"):
        expr = expr.str.replace_all(old, new, literal=True)
    return expr.str.replace_all(r"\s+", " ").str.strip_chars()


def add_normalized_text_columns(frame: pl.LazyFrame) -> pl.LazyFrame:
    return frame.with_columns(
        [
            normalize_basic_text_expr("title").alias("title_normalized"),
            normalize_basic_text_expr("description").alias("description_normalized"),
        ]
    )
