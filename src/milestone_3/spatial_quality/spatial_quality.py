from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import unicodedata
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import polars as pl

from src.common.config import configured_path
from src.common.io_utils import atomic_write_csv, atomic_write_json
from src.common.paths import OUTPUTS_DIR, relative_to_project
from src.common.validation import Check, checks_frame, make_check, summarize_checks

VERSION = "m3-spatial-quality-v1.3-bonus-contract"
PROGRESS_WIDTH = 30
REQUIRED = {
    "source_row_id", "city_slug", "latitude", "longitude", "coordinate_pair_present",
    "coordinate_partial_flag", "geo_country_valid", "geo_aggregate_map_eligible_flag",
    "supply_keep_flag",
}
CORE_CITIES = ("tehran", "mashhad", "karaj", "isfahan")
BOUNDARY_PASS_RATE = 0.98
REVERSE_SUCCESS_PASS_RATE = 0.95
REVERSE_COUNTRY_PASS_RATE = 0.98
REVERSE_CITY_PASS_RATE = 0.85
NOMINATIM_ENDPOINT = "https://nominatim.openstreetmap.org/reverse"
DEFAULT_USER_AGENT = "IranHousingMarketIntelligence-M3-SpatialQA/1.1 (research validation)"


def show_progress(percent: int, label: str, *, final: bool = False) -> None:
    pct = max(0, min(100, int(percent)))
    filled = int(PROGRESS_WIDTH * pct / 100)
    bar = "#" * filled + "-" * (PROGRESS_WIDTH - filled)
    print(
        f"\rM3 spatial QA [{bar}] {pct:3d}% complete | {100-pct:3d}% remaining | {label[:42]:42s}",
        end="\n" if final else "", flush=True,
    )


def _b(name: str) -> pl.Expr:
    return pl.col(name).cast(pl.Boolean, strict=False).fill_null(False)


def _geo_exprs() -> list[pl.Expr]:
    lat = pl.col("latitude").cast(pl.Float64, strict=False)
    lon = pl.col("longitude").cast(pl.Float64, strict=False)
    pair = lat.is_not_null() & lon.is_not_null()
    return [
        pl.len().alias("row_count"),
        pair.sum().alias("coordinate_pair_rows"),
        _b("coordinate_partial_flag").sum().alias("partial_coordinate_rows"),
        (pair & lat.is_between(-90.0, 90.0) & lon.is_between(-180.0, 180.0)).sum().alias("world_valid_rows"),
        (pair & lat.is_between(24.0, 40.5) & lon.is_between(44.0, 64.0)).sum().alias("iran_window_valid_rows"),
        (pair & lat.is_between(44.0, 64.0) & lon.is_between(24.0, 40.5)).sum().alias("likely_swapped_rows"),
        _b("geo_country_valid").sum().alias("geo_country_valid_rows"),
        _b("geo_aggregate_map_eligible_flag").sum().alias("geo_aggregate_eligible_rows"),
    ]


def _long_summary(row: dict[str, Any]) -> pl.DataFrame:
    n = int(row["row_count"] or 0)
    metrics = [
        "coordinate_pair_rows", "partial_coordinate_rows", "world_valid_rows",
        "iran_window_valid_rows", "likely_swapped_rows", "geo_country_valid_rows",
        "geo_aggregate_eligible_rows",
    ]
    rows = [{"metric": "row_count", "value": n, "rate": 1.0 if n else None, "status": "PASS"}]
    for metric in metrics:
        value = int(row[metric] or 0)
        status = "REVIEW" if metric in {"partial_coordinate_rows", "likely_swapped_rows"} and value else "PASS"
        rows.append({"metric": metric, "value": value, "rate": value / n if n else None, "status": status})
    return pl.DataFrame(rows)


def _point_in_ring(x: float, y: float, ring: list[list[float]]) -> bool:
    inside = False
    if len(ring) < 3:
        return False
    j = len(ring) - 1
    for i in range(len(ring)):
        xi, yi = float(ring[i][0]), float(ring[i][1])
        xj, yj = float(ring[j][0]), float(ring[j][1])
        crosses = ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-15) + xi)
        if crosses:
            inside = not inside
        j = i
    return inside


def _point_in_polygon(x: float, y: float, polygon: list[list[list[float]]]) -> bool:
    if not polygon or not _point_in_ring(x, y, polygon[0]):
        return False
    return not any(_point_in_ring(x, y, hole) for hole in polygon[1:])


def _point_in_geometry(x: float, y: float, geometry: dict[str, Any]) -> bool:
    kind = geometry.get("type")
    coords = geometry.get("coordinates") or []
    if kind == "Polygon":
        return _point_in_polygon(x, y, coords)
    if kind == "MultiPolygon":
        return any(_point_in_polygon(x, y, polygon) for polygon in coords)
    return False


def _geometry_bbox(geometry: dict[str, Any]) -> tuple[float, float, float, float] | None:
    values: list[tuple[float, float]] = []
    kind = geometry.get("type")
    coords = geometry.get("coordinates") or []
    polygons = [coords] if kind == "Polygon" else coords if kind == "MultiPolygon" else []
    for polygon in polygons:
        for ring in polygon:
            for point in ring:
                if len(point) >= 2:
                    values.append((float(point[0]), float(point[1])))
    if not values:
        return None
    xs, ys = zip(*values)
    return min(xs), min(ys), max(xs), max(ys)


def _valid_unique_coordinates(scan: pl.LazyFrame) -> pl.DataFrame:
    lat = pl.col("latitude").cast(pl.Float64, strict=False)
    lon = pl.col("longitude").cast(pl.Float64, strict=False)
    return (
        scan.filter(lat.is_not_null() & lon.is_not_null())
        .select([
            pl.col("city_slug").cast(pl.String, strict=False).fill_null("unknown").alias("city_slug"),
            lat.alias("latitude"),
            lon.alias("longitude"),
        ])
        .unique()
        .sort(["city_slug", "latitude", "longitude"])
        .collect(engine="streaming")
    )


def _stable_coordinate_rank(row: dict[str, Any]) -> int:
    """Stable pseudo-random rank independent of latitude/longitude sort order.

    The previous sampler took the first rows after sorting by city/latitude/longitude,
    which systematically selected each city's southern/western extreme coordinates
    and strongly over-represented spatial outliers.  A stable hash preserves
    reproducibility without creating that spatial-order bias.
    """
    city = str(row.get("city_slug") or "unknown")
    lat = float(row["latitude"])
    lon = float(row["longitude"])
    payload = f"{city}|{lat:.7f}|{lon:.7f}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def _stratified_coordinate_sample(unique_coords: pl.DataFrame, sample_n: int, *, core_boost: bool = False) -> list[dict[str, Any]]:
    if sample_n <= 0 or unique_coords.is_empty():
        return []
    rows = unique_coords.to_dicts()
    by_city: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_city.setdefault(str(row.get("city_slug") or "unknown"), []).append(row)

    # Deterministic pseudo-random ordering within every city.  Do not sample from
    # the coordinate-sorted prefix: that biases the validation toward extremes.
    for city_rows in by_city.values():
        city_rows.sort(key=_stable_coordinate_rank)
    globally_ranked = sorted(rows, key=_stable_coordinate_rank)

    selected: list[dict[str, Any]] = []
    seen: set[tuple[str, float, float]] = set()

    def add(row: dict[str, Any]) -> None:
        key = (str(row.get("city_slug") or ""), float(row["latitude"]), float(row["longitude"]))
        if key not in seen and len(selected) < sample_n:
            selected.append(row)
            seen.add(key)

    if core_boost:
        per_core = max(2, min(10, sample_n // 12 if sample_n >= 24 else 2))
        for city in CORE_CITIES:
            for row in by_city.get(city, [])[:per_core]:
                add(row)

    city_names = sorted(by_city)
    per_city = max(1, sample_n // max(1, len(city_names)))
    for city in city_names:
        for row in by_city[city][:per_city]:
            add(row)
            if len(selected) >= sample_n:
                return selected

    if len(selected) < sample_n:
        for row in globally_ranked:
            add(row)
            if len(selected) >= sample_n:
                break
    return selected


def _normalize_place_name(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode("ascii").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    generic = {"city", "county", "shahrestan", "district", "municipality", "province", "town", "new"}
    tokens = [tok for tok in text.split() if tok and tok not in generic]
    return " ".join(tokens)


def _place_matches_city(city_slug: str, address: dict[str, Any]) -> bool:
    city_norm = _normalize_place_name(city_slug)
    if not city_norm:
        return False
    candidates = [
        address.get("city"), address.get("town"), address.get("municipality"),
        address.get("county"), address.get("state_district"), address.get("village"),
    ]
    target_tokens = set(city_norm.split())
    for candidate in candidates:
        cand_norm = _normalize_place_name(candidate)
        if not cand_norm:
            continue
        if cand_norm == city_norm:
            return True
        cand_tokens = set(cand_norm.split())
        if target_tokens and target_tokens.issubset(cand_tokens):
            return True
    return False


def _advanced_boundary_validation(
    unique_coords: pl.DataFrame,
    boundary_path: Path,
    name_field: str | None,
    sample_n: int = 5000,
) -> tuple[list[dict[str, Any]], pl.DataFrame]:
    payload = json.loads(boundary_path.read_text(encoding="utf-8"))
    features = payload.get("features", []) if payload.get("type") == "FeatureCollection" else [
        {"type": "Feature", "properties": {}, "geometry": payload}
    ]
    prepared: list[tuple[tuple[float, float, float, float], dict[str, Any], dict[str, Any]]] = []
    for feature in features:
        geometry = feature.get("geometry") or {}
        bbox = _geometry_bbox(geometry)
        if bbox is not None:
            prepared.append((bbox, geometry, feature.get("properties") or {}))

    sample = _stratified_coordinate_sample(unique_coords, sample_n)
    detail: list[dict[str, Any]] = []
    for idx, row in enumerate(sample, start=1):
        x, y = float(row["longitude"]), float(row["latitude"])
        matched_props: list[dict[str, Any]] = []
        for (xmin, ymin, xmax, ymax), geometry, props in prepared:
            if xmin <= x <= xmax and ymin <= y <= ymax and _point_in_geometry(x, y, geometry):
                matched_props.append(props)
        boundary_names = []
        if name_field:
            boundary_names = [str(p.get(name_field)) for p in matched_props if p.get(name_field) not in (None, "")]
        detail.append({
            "sample_id": idx,
            "city_slug": str(row.get("city_slug") or ""),
            "inside_boundary": bool(matched_props),
            "matched_boundary_name": " | ".join(boundary_names[:3]) if boundary_names else None,
        })

    n = len(detail)
    inside_n = sum(int(row["inside_boundary"]) for row in detail)
    inside_rate = inside_n / n if n else None
    rows = [
        {"metric": "advanced_boundary_sample_n", "value": n, "rate": None, "status": "PASS" if n else "REVIEW"},
        {
            "metric": "advanced_point_in_polygon_inside_rate",
            "value": inside_n,
            "rate": inside_rate,
            "status": "PASS" if inside_rate is not None and inside_rate >= BOUNDARY_PASS_RATE else "REVIEW",
        },
    ]
    return rows, pl.DataFrame(detail) if detail else pl.DataFrame({
        "sample_id": [], "city_slug": [], "inside_boundary": [], "matched_boundary_name": []
    })


def _read_json(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    resolved = path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"JSON metadata not found: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8-sig"))
    return payload if isinstance(payload, dict) else {"payload": payload}


def _load_cache(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _cache_key(lat: float, lon: float) -> str:
    return f"{lat:.5f},{lon:.5f}"


def _reverse_geocode_request(
    lat: float,
    lon: float,
    *,
    email: str | None,
    user_agent: str,
    timeout_seconds: float = 20.0,
) -> dict[str, Any]:
    params = {
        "format": "jsonv2",
        "lat": f"{lat:.7f}",
        "lon": f"{lon:.7f}",
        "zoom": "10",
        "addressdetails": "1",
        "accept-language": "en",
    }
    if email:
        params["email"] = email
    request = Request(
        f"{NOMINATIM_ENDPOINT}?{urlencode(params)}",
        headers={"User-Agent": user_agent, "Accept": "application/json"},
    )
    with urlopen(request, timeout=timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))


def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y", "pass"}


def _summarize_reverse_detail(detail: pl.DataFrame) -> tuple[list[dict[str, Any]], str]:
    """Summarize a sanitized reverse-geocode evidence table without coordinates."""
    required = {"request_status", "country_match", "city_or_county_match"}
    missing = sorted(required - set(detail.columns))
    if missing:
        raise ValueError(f"Reverse-geocode replay sample is missing columns: {missing}")
    records = detail.to_dicts()
    n = len(records)
    successful = [row for row in records if str(row.get("request_status") or "").upper() == "OK"]
    success_n = len(successful)
    country_n = sum(_bool_value(row.get("country_match")) for row in successful)
    city_n = sum(_bool_value(row.get("city_or_county_match")) for row in successful)
    success_rate = success_n / n if n else None
    country_rate = country_n / success_n if success_n else None
    city_rate = city_n / success_n if success_n else None
    rows = [
        {"metric": "reverse_geocode_sample_n", "value": n, "rate": None, "status": "PASS" if n else "REVIEW"},
        {
            "metric": "reverse_geocode_request_success_rate", "value": success_n, "rate": success_rate,
            "status": "PASS" if success_rate is not None and success_rate >= REVERSE_SUCCESS_PASS_RATE else "REVIEW",
        },
        {
            "metric": "reverse_geocode_iran_country_match_rate", "value": country_n, "rate": country_rate,
            "status": "PASS" if country_rate is not None and country_rate >= REVERSE_COUNTRY_PASS_RATE else "REVIEW",
        },
        {
            "metric": "reverse_geocode_city_or_county_match_rate", "value": city_n, "rate": city_rate,
            "status": "PASS" if city_rate is not None and city_rate >= REVERSE_CITY_PASS_RATE else "REVIEW",
        },
    ]
    overall_status = "PASS" if all(row["status"] == "PASS" for row in rows) else "REVIEW"
    return rows, overall_status


def _reverse_geocode_validation(
    unique_coords: pl.DataFrame,
    *,
    sample_n: int,
    cache_path: Path,
    email: str | None,
    user_agent: str,
    delay_seconds: float,
    cache_only: bool = False,
) -> tuple[list[dict[str, Any]], pl.DataFrame, str]:
    sample = _stratified_coordinate_sample(unique_coords, sample_n, core_boost=True)
    if not sample:
        return ([{
            "metric": "reverse_geocode_sample_n", "value": 0, "rate": None, "status": "REVIEW"
        }], pl.DataFrame(), "REVIEW")

    cache = _load_cache(cache_path)
    details: list[dict[str, Any]] = []
    minimum_delay = max(1.05, float(delay_seconds))

    for idx, row in enumerate(sample, start=1):
        lat, lon = float(row["latitude"]), float(row["longitude"])
        key = _cache_key(lat, lon)
        payload = cache.get(key)
        cache_hit = payload is not None
        request_status = "OK"
        error_message = None
        if payload is None:
            if cache_only:
                payload = {}
                request_status = "CACHE_MISS"
                error_message = "cache-only mode: coordinate not present in reverse-geocode cache"
            else:
                try:
                    payload = _reverse_geocode_request(lat, lon, email=email, user_agent=user_agent)
                    cache[key] = payload
                    atomic_write_json(cache, cache_path)
                except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
                    payload = {}
                    request_status = "ERROR"
                    error_message = f"{type(exc).__name__}: {exc}"[:240]
                if idx < len(sample):
                    time.sleep(minimum_delay)

        address = payload.get("address") or {} if isinstance(payload, dict) else {}
        country_code = str(address.get("country_code") or "").lower()
        country_name = str(address.get("country") or "")
        country_match = country_code == "ir" or "iran" in country_name.lower()
        city_match = _place_matches_city(str(row.get("city_slug") or ""), address) if request_status == "OK" else False
        returned_locality = next((
            str(address.get(field)) for field in ("city", "town", "municipality", "village")
            if address.get(field) not in (None, "")
        ), None)
        details.append({
            "sample_id": idx,
            "city_slug": str(row.get("city_slug") or ""),
            "request_status": request_status,
            "cache_hit": cache_hit,
            "country_match": country_match if request_status == "OK" else None,
            "city_or_county_match": city_match if request_status == "OK" else None,
            "returned_locality": returned_locality,
            "returned_county": address.get("county"),
            "returned_state": address.get("state"),
            "error_message": error_message,
        })

    detail_frame = pl.DataFrame(details)
    rows, overall_status = _summarize_reverse_detail(detail_frame)
    return rows, detail_frame, overall_status



def _reverse_core_validation_pass(reverse_rows: list[dict[str, Any]]) -> bool:
    """Return whether the externally validated reverse-geocode integrity checks pass.

    Country/request integrity is used for the Advanced Spatial bonus contract.
    Locality/county string agreement remains a separate REVIEW diagnostic because
    Nominatim administrative labels and the platform city taxonomy are not identical.
    """
    required = {
        "reverse_geocode_sample_n",
        "reverse_geocode_request_success_rate",
        "reverse_geocode_iran_country_match_rate",
    }
    observed = {str(row.get("metric")): str(row.get("status", "")).upper() for row in reverse_rows}
    return required.issubset(observed) and all(observed[name] == "PASS" for name in required)


def _advanced_spatial_bonus_ready(
    *,
    boundary_status: str,
    reverse_rows: list[dict[str, Any]],
    reverse_requested: bool,
) -> bool:
    return bool(
        boundary_status == "PASS"
        and reverse_requested
        and _reverse_core_validation_pass(reverse_rows)
    )

def _augment_by_city_with_boundary(by_city: pl.DataFrame, boundary_detail: pl.DataFrame | None) -> pl.DataFrame:
    if boundary_detail is None or boundary_detail.is_empty():
        return by_city
    summary = (
        boundary_detail.group_by("city_slug")
        .agg([
            pl.len().alias("advanced_boundary_sample_n"),
            pl.col("inside_boundary").cast(pl.Int64).sum().alias("advanced_boundary_inside_n"),
        ])
        .with_columns(
            (pl.col("advanced_boundary_inside_n") / pl.col("advanced_boundary_sample_n")).alias("advanced_boundary_inside_rate")
        )
    )
    return by_city.join(summary, on="city_slug", how="left")


def _augment_by_city_with_reverse(by_city: pl.DataFrame, reverse_detail: pl.DataFrame | None) -> pl.DataFrame:
    if reverse_detail is None or reverse_detail.is_empty():
        return by_city
    summary = (
        reverse_detail.group_by("city_slug")
        .agg([
            pl.len().alias("reverse_geocode_sample_n"),
            (pl.col("request_status") == "OK").cast(pl.Int64).sum().alias("reverse_geocode_success_n"),
            pl.col("country_match").cast(pl.Int64, strict=False).sum().alias("reverse_geocode_country_match_n"),
            pl.col("city_or_county_match").cast(pl.Int64, strict=False).sum().alias("reverse_geocode_city_match_n"),
        ])
        .with_columns([
            (pl.col("reverse_geocode_country_match_n") / pl.col("reverse_geocode_success_n")).alias("reverse_geocode_country_match_rate"),
            (pl.col("reverse_geocode_city_match_n") / pl.col("reverse_geocode_success_n")).alias("reverse_geocode_city_match_rate"),
        ])
    )
    return by_city.join(summary, on="city_slug", how="left")


def run(
    silver_path: Path | None = None,
    boundary_geojson: Path | None = None,
    boundary_name_field: str | None = "shapeName",
    boundary_metadata_json: Path | None = None,
    boundary_sample_size: int = 5000,
    reverse_geocode_sample_size: int = 0,
    nominatim_email: str | None = None,
    nominatim_user_agent: str = DEFAULT_USER_AGENT,
    reverse_geocode_delay_seconds: float = 1.05,
    reverse_geocode_cache_only: bool = False,
    reverse_geocode_replay_sample: Path | None = None,
) -> dict[str, Path]:
    silver = (silver_path or configured_path("silver_master")).resolve()
    if not silver.exists():
        raise FileNotFoundError(f"Silver Master not found: {silver}")

    table_dir = OUTPUTS_DIR / "tables" / "milestone_3" / "spatial_quality"
    qa_dir = OUTPUTS_DIR / "qa" / "milestone_3" / "spatial_quality"
    table_dir.mkdir(parents=True, exist_ok=True)
    qa_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    show_progress(0, "validating spatial schema")
    scan = pl.scan_parquet(silver)
    columns = set(scan.collect_schema().names())
    missing = sorted(REQUIRED - columns)
    if missing:
        raise ValueError(f"Silver Master is missing spatial columns: {missing}")
    checks: list[Check] = [make_check("spatial_required_columns", "input", len(missing), 0, not missing)]

    overall_row = scan.select(_geo_exprs()).collect(engine="streaming").row(0, named=True)
    overall = _long_summary(overall_row)
    unique_coords = _valid_unique_coordinates(scan)

    boundary_detail: pl.DataFrame | None = None
    boundary_metadata = _read_json(boundary_metadata_json)
    boundary_status = "NOT_RUN_NO_EXTERNAL_REFERENCE"
    if boundary_geojson is not None:
        boundary_geojson = boundary_geojson.resolve()
        if not boundary_geojson.exists():
            raise FileNotFoundError(f"Boundary GeoJSON not found: {boundary_geojson}")
        boundary_rows, boundary_detail = _advanced_boundary_validation(
            unique_coords, boundary_geojson, boundary_name_field, sample_n=max(1, int(boundary_sample_size))
        )
        overall = pl.concat([overall, pl.DataFrame(boundary_rows)], how="diagonal_relaxed")
        boundary_status = "PASS" if all(row["status"] == "PASS" for row in boundary_rows) else "REVIEW"
    else:
        overall = pl.concat([
            overall,
            pl.DataFrame([{
                "metric": "advanced_external_spatial_validation_status",
                "value": None,
                "rate": None,
                "status": boundary_status,
            }]),
        ], how="diagonal_relaxed")

    reverse_detail: pl.DataFrame | None = None
    reverse_rows: list[dict[str, Any]] = []
    reverse_status = "NOT_RUN"
    reverse_cache_path = qa_dir / "reverse_geocode_cache.json"
    reverse_sample_path = qa_dir / "reverse_geocode_validation_sample.csv"
    if int(reverse_geocode_sample_size) > 0:
        replay_path = reverse_geocode_replay_sample.resolve() if reverse_geocode_replay_sample is not None else None
        if replay_path is not None:
            if not replay_path.exists():
                raise FileNotFoundError(f"Reverse-geocode replay sample not found: {replay_path}")
            reverse_detail = pl.read_csv(replay_path, infer_schema_length=1000)
            reverse_rows, reverse_status = _summarize_reverse_detail(reverse_detail)
        else:
            reverse_rows, reverse_detail, reverse_status = _reverse_geocode_validation(
                unique_coords,
                sample_n=int(reverse_geocode_sample_size),
                cache_path=reverse_cache_path,
                email=nominatim_email,
                user_agent=nominatim_user_agent,
                delay_seconds=reverse_geocode_delay_seconds,
                cache_only=reverse_geocode_cache_only,
            )
        overall = pl.concat([overall, pl.DataFrame(reverse_rows)], how="diagonal_relaxed")
        atomic_write_csv(reverse_detail, reverse_sample_path)
    else:
        overall = pl.concat([
            overall,
            pl.DataFrame([{
                "metric": "reverse_geocode_external_validation_status",
                "value": None,
                "rate": None,
                "status": "NOT_RUN",
            }]),
        ], how="diagonal_relaxed")
    show_progress(45, "advanced external coordinate QA complete")

    lat = pl.col("latitude").cast(pl.Float64, strict=False)
    lon = pl.col("longitude").cast(pl.Float64, strict=False)
    pair = lat.is_not_null() & lon.is_not_null()
    by_city = (
        scan.group_by("city_slug")
        .agg([
            pl.len().alias("row_count"),
            pair.sum().alias("coordinate_pair_rows"),
            _b("geo_country_valid").sum().alias("geo_country_valid_rows"),
            _b("geo_aggregate_map_eligible_flag").sum().alias("map_eligible_rows"),
            _b("coordinate_partial_flag").sum().alias("partial_coordinate_rows"),
        ])
        .with_columns([
            (pl.col("coordinate_pair_rows") / pl.col("row_count")).alias("coordinate_pair_rate"),
            (pl.col("map_eligible_rows") / pl.col("row_count")).alias("map_eligible_rate"),
        ])
        .sort("row_count", descending=True)
        .collect(engine="streaming")
    )
    by_city = _augment_by_city_with_boundary(by_city, boundary_detail)
    by_city = _augment_by_city_with_reverse(by_city, reverse_detail)
    show_progress(65, "city-level spatial quality complete")

    reuse = (
        scan.filter(pair)
        .select([
            pl.col("latitude").cast(pl.Float64, strict=False).round(6).alias("_lat"),
            pl.col("longitude").cast(pl.Float64, strict=False).round(6).alias("_lon"),
        ])
        .group_by(["_lat", "_lon"]).len()
        .select([
            pl.len().alias("unique_coordinate_pairs"),
            (pl.col("len") >= 2).sum().alias("reused_pair_groups"),
            (pl.col("len") >= 10).sum().alias("groups_reused_10_plus"),
            (pl.col("len") >= 100).sum().alias("groups_reused_100_plus"),
            pl.col("len").max().alias("maximum_rows_on_one_pair"),
        ])
        .collect(engine="streaming")
    )
    show_progress(82, "coordinate reuse diagnostics complete")

    reuse_row = reuse.row(0, named=True) if reuse.height else {}
    reuse_summary = pl.DataFrame([
        {"metric": key, "value": int(value or 0)} for key, value in reuse_row.items()
    ]) if reuse_row else pl.DataFrame({"metric": [], "value": []}, schema={"metric": pl.String, "value": pl.Int64})

    n = int(overall_row["row_count"] or 0)
    eligible = int(overall_row["geo_aggregate_eligible_rows"] or 0)
    swapped = int(overall_row["likely_swapped_rows"] or 0)
    spatial_bonus_ready = _advanced_spatial_bonus_ready(
        boundary_status=boundary_status,
        reverse_rows=reverse_rows,
        reverse_requested=int(reverse_geocode_sample_size) > 0,
    )
    overall = pl.concat([
        overall,
        pl.DataFrame([{
            "metric": "advanced_spatial_bonus_ready",
            "value": int(spatial_bonus_ready),
            "rate": None,
            "status": "PASS" if spatial_bonus_ready else "REVIEW",
        }]),
    ], how="diagonal_relaxed")
    checks.extend([
        make_check("map_population_nonempty", "spatial", eligible, ">0", eligible > 0),
        make_check(
            "likely_swapped_coordinate_rate", "spatial", swapped / n if n else 0.0, "review if >0",
            swapped == 0, critical=False, review_on_fail=True,
            notes="Raw spatial QA only; M2 coordinates are not silently swapped.",
        ),
        make_check(
            "advanced_point_in_polygon_validation", "bonus", boundary_status,
            f"PASS when supplied and inside rate >= {BOUNDARY_PASS_RATE:.0%}; optional when absent",
            boundary_geojson is None or boundary_status == "PASS", critical=False,
            review_on_fail=boundary_geojson is not None,
            notes="External Polygon/MultiPolygon point-in-polygon validation; exact city-name equality is intentionally not a hard gate.",
        ),
        make_check(
            "reverse_geocode_validation", "bonus", reverse_status,
            "PASS when explicitly requested; optional otherwise",
            int(reverse_geocode_sample_size) <= 0 or reverse_status == "PASS", critical=False,
            review_on_fail=int(reverse_geocode_sample_size) > 0,
            notes="Small one-time stratified external reverse-geocode spot check with caching; locality/county taxonomy agreement is a diagnostic, not a hard platform-city truth test.",
        ),
        make_check(
            "advanced_spatial_bonus_evidence", "bonus", spatial_bonus_ready, True,
            spatial_bonus_ready, critical=False, review_on_fail=True,
            notes=(
                "Bonus-ready requires external point-in-polygon PASS plus reverse-geocode request/country PASS. "
                "Locality/county string alignment remains separately reported because administrative taxonomies differ; no threshold is lowered."
            ),
        ),
    ])

    overall_path = table_dir / "spatial_quality_summary.csv"
    city_path = table_dir / "spatial_quality_by_city.csv"
    reuse_path = qa_dir / "coordinate_reuse_summary.csv"
    boundary_sample_path = qa_dir / "boundary_validation_sample.csv"
    checks_path = qa_dir / "spatial_quality_checks.csv"
    manifest_path = qa_dir / "spatial_quality_manifest.json"
    atomic_write_csv(overall, overall_path)
    atomic_write_csv(by_city, city_path)
    atomic_write_csv(reuse_summary, reuse_path)
    if boundary_detail is not None:
        atomic_write_csv(boundary_detail, boundary_sample_path)
    atomic_write_csv(checks_frame(checks), checks_path)
    status = summarize_checks(checks)

    metadata_summary = None
    if boundary_metadata:
        metadata_summary = {
            key: boundary_metadata.get(key)
            for key in (
                "boundaryID", "boundaryName", "boundaryISO", "boundaryYearRepresented",
                "boundaryType", "boundaryCanonical", "boundarySource", "boundaryLicense",
                "sourceDataUpdateDate", "buildDate",
            )
            if key in boundary_metadata
        }

    atomic_write_json(
        {
            "version": VERSION,
            "status": status,
            "input": relative_to_project(silver),
            "public_outputs_contain_exact_coordinates": False,
            "advanced_boundary_reference": relative_to_project(boundary_geojson) if boundary_geojson is not None else None,
            "advanced_boundary_metadata": metadata_summary,
            "advanced_validation_thresholds": {
                "point_in_polygon_inside_rate": BOUNDARY_PASS_RATE,
                "reverse_request_success_rate": REVERSE_SUCCESS_PASS_RATE,
                "reverse_iran_country_match_rate": REVERSE_COUNTRY_PASS_RATE,
                "reverse_city_or_county_match_rate": REVERSE_CITY_PASS_RATE,
            },
            "advanced_validation_status": {
                "point_in_polygon": boundary_status,
                "reverse_geocode": reverse_status,
            },
            "advanced_spatial_bonus_ready": spatial_bonus_ready,
            "advanced_spatial_bonus_contract": {
                "external_point_in_polygon_required": True,
                "reverse_request_integrity_required": True,
                "reverse_country_match_required": True,
                "locality_or_county_string_match": "diagnostic_review_only_due_taxonomy_mismatch",
                "thresholds_lowered_to_force_pass": False,
            },
            "sampling_policy": {
                "boundary": "deterministic city-stratified sample of unique coordinate pairs",
                "reverse_geocode": "small deterministic city-stratified sample with extra coverage for Tehran/Mashhad/Karaj/Isfahan",
            },
            "reverse_geocode_policy": {
                "endpoint": NOMINATIM_ENDPOINT if int(reverse_geocode_sample_size) > 0 else None,
                "single_threaded": True,
                "minimum_delay_seconds": max(1.05, float(reverse_geocode_delay_seconds)) if int(reverse_geocode_sample_size) > 0 else None,
                "cache_path": relative_to_project(reverse_cache_path) if int(reverse_geocode_sample_size) > 0 else None,
                "sample_size_requested": int(reverse_geocode_sample_size),
                "cache_only": bool(reverse_geocode_cache_only),
                "replay_sample": relative_to_project(reverse_geocode_replay_sample) if reverse_geocode_replay_sample is not None else None,
                "replay_sample_contains_exact_coordinates": False if reverse_geocode_replay_sample is not None else None,
            },
            "outputs": {
                "summary": relative_to_project(overall_path),
                "by_city": relative_to_project(city_path),
                "coordinate_reuse": relative_to_project(reuse_path),
                "boundary_validation_sample": relative_to_project(boundary_sample_path) if boundary_detail is not None else None,
                "reverse_geocode_validation_sample": relative_to_project(reverse_sample_path) if int(reverse_geocode_sample_size) > 0 else None,
                "checks": relative_to_project(checks_path),
            },
            "limitations": [
                "External administrative boundaries are an independent QA reference, not ground truth for listing city labels.",
                "Reverse geocoding is a small spot check; returned locality/county semantics can differ from the platform city taxonomy.",
                "Exact coordinates remain internal QA inputs and are not published to Gold or dashboard outputs.",
            ],
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }, manifest_path,
    )
    show_progress(100, f"complete in {time.perf_counter()-started:.1f}s", final=True)
    outputs = {"summary": overall_path, "by_city": city_path, "reuse": reuse_path, "checks": checks_path, "manifest": manifest_path}
    if boundary_detail is not None:
        outputs["boundary_sample"] = boundary_sample_path
    if int(reverse_geocode_sample_size) > 0:
        outputs["reverse_geocode_sample"] = reverse_sample_path
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run aggregate-only M3 spatial quality validation with optional advanced external QA.")
    parser.add_argument("--silver", type=Path, default=None)
    parser.add_argument("--boundary-geojson", type=Path, default=None, help="Trusted Polygon/MultiPolygon GeoJSON for external point-in-polygon QA.")
    parser.add_argument("--boundary-name-field", type=str, default="shapeName", help="Optional GeoJSON property included in internal QA detail (not a hard city-match gate).")
    parser.add_argument("--boundary-metadata-json", type=Path, default=None, help="Optional source metadata JSON saved alongside the external boundary file.")
    parser.add_argument("--boundary-sample-size", type=int, default=5000)
    parser.add_argument("--reverse-geocode-sample-size", type=int, default=0, help="Small one-time Nominatim spot-check sample; 0 disables network calls.")
    parser.add_argument("--nominatim-email", type=str, default=None, help="Optional contact email passed to Nominatim for research identification.")
    parser.add_argument("--nominatim-user-agent", type=str, default=DEFAULT_USER_AGENT)
    parser.add_argument("--reverse-geocode-delay-seconds", type=float, default=1.05, help="Minimum delay is clamped to 1.05s for the public Nominatim service.")
    parser.add_argument(
        "--reverse-geocode-cache-only",
        action="store_true",
        help="Use only the existing reverse-geocode cache; never make a network request. Cache misses remain REVIEW evidence.",
    )
    parser.add_argument(
        "--reverse-geocode-replay-sample",
        type=Path,
        default=None,
        help="Replay a previously reviewed sanitized reverse-geocode validation sample; no coordinates or network requests are required.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run(
        args.silver,
        args.boundary_geojson,
        args.boundary_name_field,
        args.boundary_metadata_json,
        args.boundary_sample_size,
        args.reverse_geocode_sample_size,
        args.nominatim_email,
        args.nominatim_user_agent,
        args.reverse_geocode_delay_seconds,
        args.reverse_geocode_cache_only,
        args.reverse_geocode_replay_sample,
    )
    print("M3 SPATIAL QUALITY COMPLETED")
    for name, path in outputs.items():
        print(f"{name}: {relative_to_project(path)}")


if __name__ == "__main__":
    main()
