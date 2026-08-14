from __future__ import annotations

from datetime import date

import polars as pl

from src.common.config import setting
from src.common.domain import property_family_categories

MISSING_TEXT = ["", "null", "<null>", "none", "nan", "n/a", "na"]
PERSIAN_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
ARABIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"
ASCII_DIGITS = "0123456789"

BOOLEAN_COLUMNS = [
    "rent_to_single", "transformable_price", "rent_credit_transform",
    "has_business_deed", "has_elevator", "has_warehouse", "has_parking",
    "is_rebuilt", "has_water", "has_electricity", "has_gas",
    "has_security_guard", "has_barbecue", "has_pool", "has_jacuzzi",
    "has_sauna", "has_balcony",
]

MONEY_COLUMNS = {
    "price_value": "price_value_toman",
    "credit_value": "credit_value_toman",
    "rent_value": "rent_value_toman",
    "transformable_credit": "transformable_credit_toman",
    "transformed_credit": "transformed_credit_toman",
    "transformable_rent": "transformable_rent_toman",
    "transformed_rent": "transformed_rent_toman",
    "cost_per_extra_person": "cost_per_extra_person_toman",
    "rent_price_on_regular_days": "rent_price_on_regular_days_toman",
    "rent_price_at_weekends": "rent_price_at_weekends_toman",
    "rent_price_on_special_days": "rent_price_on_special_days_toman",
}


def normalized_text(column: str) -> pl.Expr:
    value = pl.col(column).cast(pl.String).str.strip_chars()
    missing = value.is_null() | value.str.to_lowercase().is_in(MISSING_TEXT)
    return pl.when(missing).then(pl.lit(None, dtype=pl.String)).otherwise(value)


def normalized_number_text(column: str) -> pl.Expr:
    value = normalized_text(column)
    for source, target in zip(PERSIAN_DIGITS + ARABIC_DIGITS, ASCII_DIGITS * 2):
        value = value.str.replace_all(source, target, literal=True)
    return (
        value.str.replace_all(",", "", literal=True)
        .str.replace_all("٬", "", literal=True)
        .str.replace_all("_", "", literal=True)
        .str.replace_all(" ", "", literal=True)
    )


def numeric_expr(column: str, dtype: pl.DataType) -> pl.Expr:
    return normalized_number_text(column).cast(dtype, strict=False)


def boolean_expr(column: str) -> pl.Expr:
    value = normalized_text(column).str.to_lowercase()
    return (
        pl.when(value == "true").then(pl.lit(True))
        .when(value == "false").then(pl.lit(False))
        .otherwise(pl.lit(None, dtype=pl.Boolean))
    )


def rooms_expr() -> pl.Expr:
    value = normalized_text("rooms_count")
    return (
        pl.when(value == "بدون اتاق").then(pl.lit(0))
        .when(value == "یک").then(pl.lit(1))
        .when(value == "دو").then(pl.lit(2))
        .when(value == "سه").then(pl.lit(3))
        .when(value == "چهار").then(pl.lit(4))
        .when(value == "پنج یا بیشتر").then(pl.lit(5))
        .otherwise(pl.lit(None, dtype=pl.Int8))
        .cast(pl.Int8)
    )


def capped_integer_expr(column: str, marker: str, lower_bound: int) -> pl.Expr:
    value = normalized_text(column)
    return (
        pl.when(value == marker).then(pl.lit(lower_bound))
        .otherwise(numeric_expr(column, pl.Int16))
        .cast(pl.Int16)
    )


def _property_family_expr() -> pl.Expr:
    families = property_family_categories()
    expr = pl.when(pl.lit(False)).then(pl.lit("other"))
    for family, categories in families.items():
        if family == "other" or not categories:
            continue
        expr = expr.when(pl.col("cat3_slug").is_in(sorted(categories))).then(pl.lit(family))
    return expr.otherwise(pl.lit("other")).alias("property_family")


def apply_structured_standardization(frame: pl.LazyFrame) -> pl.LazyFrame:
    """Type and standardize Raw fields while preserving every Raw source column."""
    names = frame.collect_schema().names()
    if "source_row_id" not in names:
        frame = frame.with_row_index("source_row_id", offset=1)

    date_text = normalized_text("created_at_month")
    analysis_month = pl.coalesce(
        [
            date_text.str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S", strict=False).dt.date(),
            date_text.str.strptime(pl.Date, format="%Y-%m-%d", strict=False),
        ]
    )

    typed: list[pl.Expr] = [
        analysis_month.alias("analysis_month"),
        numeric_expr("building_size", pl.Float64).alias("building_size_sqm"),
        numeric_expr("land_size", pl.Float64).alias("land_size_sqm"),
        numeric_expr("location_latitude", pl.Float64).alias("latitude"),
        numeric_expr("location_longitude", pl.Float64).alias("longitude"),
        numeric_expr("location_radius", pl.Float64).alias("location_radius_m"),
        numeric_expr("regular_person_capacity", pl.Int16).alias("regular_person_capacity_num"),
        numeric_expr("extra_person_capacity", pl.Int16).alias("extra_person_capacity_num"),
        rooms_expr().alias("rooms_count_num"),
        numeric_expr("construction_year", pl.Int16).alias("construction_year_jalali"),
        capped_integer_expr("floor", "30+", 30).alias("floor_num"),
        capped_integer_expr("total_floors_count", "30+", 30).alias("total_floors_count_num"),
        capped_integer_expr("unit_per_floor", "more_than_8", 9).alias("unit_per_floor_num"),
        (normalized_text("rooms_count") == "پنج یا بیشتر").fill_null(False).alias("rooms_count_censored_flag"),
        (normalized_text("construction_year") == "قبل از ۱۳۷۰").fill_null(False).alias("construction_year_before_1370_flag"),
        (normalized_text("floor") == "30+").fill_null(False).alias("floor_censored_flag"),
        (normalized_text("total_floors_count") == "30+").fill_null(False).alias("total_floors_count_censored_flag"),
        (normalized_text("total_floors_count") == "unselect").fill_null(False).alias("total_floors_count_unselected_flag"),
        (normalized_text("unit_per_floor") == "more_than_8").fill_null(False).alias("unit_per_floor_censored_flag"),
        (normalized_text("unit_per_floor") == "unselect").fill_null(False).alias("unit_per_floor_unselected_flag"),
        (normalized_text("has_balcony") == "unselect").fill_null(False).alias("has_balcony_unselected_flag"),
        _property_family_expr(),
    ]
    typed.extend(numeric_expr(source, pl.Int64).alias(target) for source, target in MONEY_COLUMNS.items())
    typed.extend(boolean_expr(column).alias(f"{column}_bool") for column in BOOLEAN_COLUMNS)
    frame = frame.with_columns(typed)

    parse_pairs = [
        ("created_at_month", "analysis_month"), ("building_size", "building_size_sqm"),
        ("land_size", "land_size_sqm"), ("location_latitude", "latitude"),
        ("location_longitude", "longitude"), ("location_radius", "location_radius_m"),
        ("regular_person_capacity", "regular_person_capacity_num"),
        ("extra_person_capacity", "extra_person_capacity_num"),
        ("rooms_count", "rooms_count_num"), ("floor", "floor_num"),
        *MONEY_COLUMNS.items(), *[(c, f"{c}_bool") for c in BOOLEAN_COLUMNS],
    ]
    parse_errors = [
        (
            normalized_text(source).is_not_null()
            & pl.col(target).is_null()
            & ~((pl.lit(source) == "has_balcony") & (normalized_text(source) == "unselect"))
        ).cast(pl.UInt8)
        for source, target in parse_pairs
    ]
    parse_errors.extend(
        [
            (normalized_text("construction_year").is_not_null() & pl.col("construction_year_jalali").is_null() & (normalized_text("construction_year") != "قبل از ۱۳۷۰")).cast(pl.UInt8),
            (normalized_text("total_floors_count").is_not_null() & pl.col("total_floors_count_num").is_null() & (normalized_text("total_floors_count") != "unselect")).cast(pl.UInt8),
            (normalized_text("unit_per_floor").is_not_null() & pl.col("unit_per_floor_num").is_null() & (normalized_text("unit_per_floor") != "unselect")).cast(pl.UInt8),
        ]
    )
    frame = frame.with_columns(pl.sum_horizontal(parse_errors).cast(pl.UInt16).alias("type_parse_error_count"))

    core_start = date.fromisoformat(str(setting("analysis", "core_period", "start", default="2024-05-01")))
    core_end = date.fromisoformat(str(setting("analysis", "core_period", "end", default="2024-12-31")))
    families = property_family_categories()
    standard_building = sorted(families.get("apartment", set()) | families.get("commercial", set()))
    villa = sorted(families.get("villa", set()))
    industrial = sorted(families.get("industrial", set()))
    land = sorted(families.get("land", set()))

    frame = frame.with_columns(
        [
            pl.col("analysis_month").is_between(pl.lit(core_start), pl.lit(core_end), closed="both").fill_null(False).alias("is_core_analysis_period"),
            pl.col("cat2_slug").is_in(["residential-sell", "commercial-sell"]).fill_null(False).alias("is_sale"),
            pl.col("cat2_slug").is_in(["residential-rent", "commercial-rent"]).fill_null(False).alias("is_long_term_rent"),
            (pl.col("cat2_slug") == "temporary-rent").fill_null(False).alias("is_temporary_rent"),
            pl.when(pl.col("cat2_slug").is_in(["residential-sell", "commercial-sell"]))
            .then(pl.lit("sale"))
            .when(pl.col("cat2_slug").is_in(["residential-rent", "commercial-rent"]))
            .then(pl.lit("long_term_rent"))
            .when(pl.col("cat2_slug") == "temporary-rent").then(pl.lit("temporary_rent"))
            .when(pl.col("cat2_slug") == "real-estate-services").then(pl.lit("service"))
            .otherwise(pl.lit("unknown")).alias("market_regime"),
            (pl.col("latitude").is_not_null() & pl.col("longitude").is_not_null()).alias("coordinate_pair_present"),
            (pl.col("latitude").is_null() != pl.col("longitude").is_null()).alias("coordinate_partial_flag"),
            pl.when(pl.col("cat3_slug").is_in(standard_building)).then(pl.lit("building_area"))
            .when(pl.col("cat3_slug").is_in(land)).then(pl.lit("source_building_size_as_parcel_area"))
            .when(pl.col("cat3_slug").is_in(villa)).then(pl.lit("building_and_land_separate"))
            .when(pl.col("cat3_slug").is_in(industrial)).then(pl.lit("heterogeneous_manual_review"))
            .when(pl.col("cat2_slug") == "temporary-rent").then(pl.lit("temporary_rental_area"))
            .otherwise(pl.lit("not_applicable_or_unknown")).alias("area_basis"),
            pl.when(pl.col("cat3_slug").is_in(standard_building + land)).then(pl.col("building_size_sqm"))
            .otherwise(pl.lit(None, dtype=pl.Float64)).alias("primary_area_sqm"),
            (pl.col("latitude").is_not_null() & pl.col("longitude").is_not_null() & pl.col("latitude").is_between(24.0, 40.5, closed="both") & pl.col("longitude").is_between(43.0, 64.5, closed="both")).fill_null(False).alias("geo_country_valid"),
        ]
    )

    frame = frame.with_columns(
        [
            pl.when(~pl.col("is_sale")).then(pl.lit("not_applicable"))
            .when((pl.col("price_mode") == "مقطوع") & (pl.col("price_value_toman") > 0)).then(pl.lit("fixed_positive"))
            .when((pl.col("price_mode") == "مجانی") & (pl.col("price_value_toman") == 0)).then(pl.lit("semantic_zero_not_sale_price"))
            .when(pl.col("price_mode") == "توافقی").then(pl.lit("negotiable"))
            .when(pl.col("price_mode").is_null()).then(pl.lit("missing"))
            .otherwise(pl.lit("inconsistent")).alias("sale_price_status"),
            pl.when(pl.col("is_sale") & (pl.col("price_mode") == "مقطوع") & (pl.col("price_value_toman") > 0)).then(pl.col("price_value_toman"))
            .otherwise(pl.lit(None, dtype=pl.Int64)).alias("sale_price_clean_toman"),
            pl.when(~pl.col("is_long_term_rent")).then(pl.lit("not_applicable"))
            .when((pl.col("rent_mode") == "مقطوع") & (pl.col("rent_value_toman") > 0)).then(pl.lit("fixed_positive"))
            .when((pl.col("rent_mode") == "مجانی") & (pl.col("rent_value_toman") == 0)).then(pl.lit("semantic_zero"))
            .when(pl.col("rent_mode") == "توافقی").then(pl.lit("negotiable"))
            .when(pl.col("rent_mode").is_null()).then(pl.lit("missing"))
            .otherwise(pl.lit("inconsistent")).alias("rent_component_status"),
            pl.when(~pl.col("is_long_term_rent")).then(pl.lit("not_applicable"))
            .when((pl.col("credit_mode") == "مقطوع") & (pl.col("credit_value_toman") > 0)).then(pl.lit("fixed_positive"))
            .when((pl.col("credit_mode") == "مجانی") & (pl.col("credit_value_toman") == 0)).then(pl.lit("semantic_zero"))
            .when(pl.col("credit_mode") == "توافقی").then(pl.lit("negotiable"))
            .when(pl.col("credit_mode").is_null()).then(pl.lit("missing"))
            .otherwise(pl.lit("inconsistent")).alias("credit_component_status"),
        ]
    )

    frame = frame.with_columns(
        [
            pl.when(~pl.col("is_long_term_rent")).then(pl.lit("not_applicable"))
            .when((pl.col("rent_component_status") == "fixed_positive") & (pl.col("credit_component_status") == "fixed_positive")).then(pl.lit("rent_plus_deposit"))
            .when((pl.col("rent_component_status") == "semantic_zero") & (pl.col("credit_component_status") == "fixed_positive")).then(pl.lit("full_deposit"))
            .when((pl.col("rent_component_status") == "fixed_positive") & (pl.col("credit_component_status") == "semantic_zero")).then(pl.lit("rent_only"))
            .when((pl.col("rent_component_status") == "negotiable") | (pl.col("credit_component_status") == "negotiable")).then(pl.lit("negotiable"))
            .otherwise(pl.lit("unknown_or_incomplete")).alias("long_term_rent_regime"),
            pl.when(pl.col("is_sale") & pl.col("sale_price_clean_toman").is_not_null() & (pl.col("primary_area_sqm") > 0))
            .then(pl.col("sale_price_clean_toman") / pl.col("primary_area_sqm"))
            .otherwise(pl.lit(None, dtype=pl.Float64)).alias("price_per_sqm_raw_toman"),
        ]
    )

    listing_jalali_year = pl.when(pl.col("analysis_month").dt.month() >= 4).then(pl.col("analysis_month").dt.year() - 621).otherwise(pl.col("analysis_month").dt.year() - 622)
    year_invalid = (
        pl.col("construction_year_jalali").is_not_null()
        & ((pl.col("construction_year_jalali") < 1370) | (pl.col("construction_year_jalali") > listing_jalali_year))
    ).fill_null(False)
    frame = frame.with_columns(
        [
            year_invalid.alias("construction_year_numeric_invalid_flag"),
            pl.when(pl.col("construction_year_jalali").is_not_null() & ~year_invalid)
            .then((listing_jalali_year - pl.col("construction_year_jalali")).cast(pl.Int16))
            .otherwise(pl.lit(None, dtype=pl.Int16)).alias("building_age_years"),
        ]
    )
    return frame


def write_standardization_summary(input_path: "Path", output_path: "Path | None" = None) -> "Path":
    from pathlib import Path
    from src.common.io_utils import atomic_write_csv
    from src.common.paths import OUTPUTS_DIR

    output_path = output_path or OUTPUTS_DIR / "tables" / "milestone_2" / "standardization" / "standardization_summary.csv"
    scan = pl.scan_parquet(input_path)
    values = scan.select([
        pl.len().alias("row_count"),
        (pl.col("type_parse_error_count") > 0).sum().alias("rows_with_parse_error"),
        pl.col("analysis_month").is_null().sum().alias("analysis_month_parse_failure_rows"),
        (pl.col("title").fill_null("") != pl.col("title_normalized").fill_null("")).sum().alias("title_normalized_changed_rows"),
        (pl.col("description").fill_null("") != pl.col("description_normalized").fill_null("")).sum().alias("description_normalized_changed_rows"),
    ]).collect(engine="streaming").row(0, named=True)
    review_metrics = {"rows_with_parse_error", "analysis_month_parse_failure_rows"}
    rows = [
        {"metric": key, "value": str(value), "status": "REVIEW" if key in review_metrics and int(value or 0) else "PASS"}
        for key, value in values.items()
    ]
    atomic_write_csv(pl.DataFrame(rows), output_path)
    return output_path
