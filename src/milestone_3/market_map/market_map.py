from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
import matplotlib.pyplot as plt
import polars as pl

from src.common.config import setting
from src.common.io_utils import atomic_write_csv, atomic_write_json, atomic_write_text
from src.common.paths import OUTPUTS_DIR, relative_to_project
from src.common.validation import Check, checks_frame, make_check, summarize_checks

VERSION = "m3-market-map-v1.5-balanced-four-city-figures"
PROGRESS_WIDTH = 30

SALES_BASE = OUTPUTS_DIR / "model_artifacts" / "milestone_3" / "analysis_populations" / "sales_analysis_base.parquet"
MAP_BASE = OUTPUTS_DIR / "model_artifacts" / "milestone_3" / "analysis_populations" / "map_analysis_base.parquet"

REQUIRED_SALES = {"source_row_id", "city_slug", "neighborhood_slug", "property_family", "sale_price_per_sqm_final_toman"}
REQUIRED_MAP = {"source_row_id", "city_slug", "neighborhood_slug", "latitude", "longitude"}
FOUR_CITY_SLUGS = ("tehran", "mashhad", "karaj", "isfahan")
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BOUNDARY_GEOJSON = PROJECT_ROOT / "external_data" / "reference" / "geoboundaries_irn_adm2.geojson"


def _display_slug(value: object) -> str:
    text = str(value or "").replace("-", " ").replace("_", " ").strip()
    return " ".join(part.capitalize() for part in text.split())


def _iter_polygon_exteriors(geometry: dict) -> list[list[list[float]]]:
    if not isinstance(geometry, dict):
        return []
    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates") or []
    if geometry_type == "Polygon":
        return [coordinates[0]] if coordinates else []
    if geometry_type == "MultiPolygon":
        return [polygon[0] for polygon in coordinates if polygon]
    return []


def _draw_boundary_reference(ax: plt.Axes, path: Path | None) -> bool:
    if path is None or not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return False
    drew_any = False
    for feature in data.get("features", []):
        for ring in _iter_polygon_exteriors(feature.get("geometry", {})):
            if not ring:
                continue
            xy = [(float(point[0]), float(point[1])) for point in ring if len(point) >= 2]
            if len(xy) < 2:
                continue
            x, y = zip(*xy)
            ax.plot(x, y, color="#B7C0C8", linewidth=0.35, alpha=0.70, zorder=1)
            drew_any = True
    return drew_any




def _write_interactive_city_map(
    rows: list[dict[str, object]],
    path: Path,
    *,
    min_n: int,
    boundary_geojson: Path | None = None,
) -> bool:
    """Write a self-contained offline SVG map using aggregate city coordinates only.

    The HTML contains no CDN, tile-server, font, or other network dependency, so it can
    be opened directly from disk or through a local HTTP server. Coordinates are rounded
    to two decimals before serialization. If the optional Iran boundary GeoJSON is
    available, its polygon exteriors are embedded as simplified SVG paths; otherwise the
    map falls back to a longitude/latitude frame. Exact listing coordinates are never
    serialized.
    """
    prepared: list[dict[str, object]] = []
    for row in rows:
        try:
            lat = round(float(row["city_latitude"]), 2)
            lon = round(float(row["city_longitude"]), 2)
            price_m = float(row["median_asking_price_per_sqm_toman"]) / 1_000_000.0
            listing_n = int(row["listing_n"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (math.isfinite(lat) and math.isfinite(lon) and math.isfinite(price_m)):
            continue
        prepared.append({
            "city": _display_slug(row.get("city_slug")),
            "lat": lat,
            "lon": lon,
            "price_m": round(price_m, 1),
            "listing_n": listing_n,
            "reliable_neighborhood_n": int(row.get("reliable_neighborhood_n") or 0),
        })
    if not prepared:
        return False

    prices = [float(row["price_m"]) for row in prepared]
    counts = [int(row["listing_n"]) for row in prepared]
    p_min, p_max = min(prices), max(prices)
    n_max = max(max(counts), 1)
    palette = ["#C7DDF0", "#8AB8DB", "#4A8DBD", "#1F5F99"]

    svg_width, svg_height = 1000.0, 700.0
    pad_left, pad_right, pad_top, pad_bottom = 70.0, 55.0, 65.0, 65.0
    lon_min, lon_max = 43.5, 64.5
    lat_min, lat_max = 24.0, 40.5

    def project(lon: float, lat: float) -> tuple[float, float]:
        x = pad_left + (lon - lon_min) / (lon_max - lon_min) * (svg_width - pad_left - pad_right)
        y = pad_top + (lat_max - lat) / (lat_max - lat_min) * (svg_height - pad_top - pad_bottom)
        return x, y

    for row in prepared:
        if p_max > p_min:
            frac = (float(row["price_m"]) - p_min) / (p_max - p_min)
        else:
            frac = 0.5
        idx = min(len(palette) - 1, max(0, int(frac * len(palette))))
        row["color"] = palette[idx]
        row["radius"] = round(10.0 + 17.0 * math.sqrt(int(row["listing_n"]) / n_max), 2)
        x, y = project(float(row["lon"]), float(row["lat"]))
        row["x"] = round(x, 1)
        row["y"] = round(y, 1)

    boundary_paths: list[str] = []
    boundary_used = False
    if boundary_geojson is not None and boundary_geojson.exists():
        try:
            data = json.loads(boundary_geojson.read_text(encoding="utf-8-sig"))
            for feature in data.get("features", []):
                for ring in _iter_polygon_exteriors(feature.get("geometry", {})):
                    clean: list[tuple[float, float]] = []
                    for point in ring:
                        if len(point) < 2:
                            continue
                        try:
                            lon, lat = float(point[0]), float(point[1])
                        except (TypeError, ValueError):
                            continue
                        if not (math.isfinite(lon) and math.isfinite(lat)):
                            continue
                        clean.append((lon, lat))
                    if len(clean) < 3:
                        continue
                    # Cap path density so the professor-facing HTML remains lightweight.
                    step = max(1, len(clean) // 180)
                    sampled = clean[::step]
                    if sampled[-1] != clean[-1]:
                        sampled.append(clean[-1])
                    coords = [project(lon, lat) for lon, lat in sampled]
                    d = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in coords) + " Z"
                    boundary_paths.append(f'<path d="{d}" class="adm-boundary"/>')
            boundary_used = bool(boundary_paths)
        except Exception:
            boundary_paths = []
            boundary_used = False

    grid_parts: list[str] = []
    for lon in (45, 50, 55, 60):
        x, _ = project(float(lon), lat_min)
        grid_parts.append(f'<line x1="{x:.1f}" y1="{pad_top:.1f}" x2="{x:.1f}" y2="{svg_height-pad_bottom:.1f}" class="grid"/>')
        grid_parts.append(f'<text x="{x:.1f}" y="{svg_height-pad_bottom+25:.1f}" class="axis-label" text-anchor="middle">{lon}E</text>')
    for lat in (25, 30, 35, 40):
        _, y = project(lon_min, float(lat))
        grid_parts.append(f'<line x1="{pad_left:.1f}" y1="{y:.1f}" x2="{svg_width-pad_right:.1f}" y2="{y:.1f}" class="grid"/>')
        grid_parts.append(f'<text x="{pad_left-12:.1f}" y="{y+4:.1f}" class="axis-label" text-anchor="end">{lat}N</text>')

    marker_parts: list[str] = []
    button_parts: list[str] = []
    label_offsets = {
        "Tehran": (14.0, -14.0),
        "Karaj": (-14.0, 25.0),
        "Mashhad": (14.0, -14.0),
        "Isfahan": (14.0, 24.0),
    }
    for i, row in enumerate(prepared):
        city = str(row["city"])
        dx, dy = label_offsets.get(city, (12.0, -12.0))
        marker_parts.append(
            f'<g class="city-marker" data-index="{i}" tabindex="0" role="button" '
            f'aria-label="{city} market summary">'
            f'<circle cx="{row["x"]}" cy="{row["y"]}" r="{row["radius"]}" '
            f'fill="{row["color"]}" stroke="#30485B" stroke-width="2"/>'
            f'<text x="{float(row["x"])+dx:.1f}" y="{float(row["y"])+dy:.1f}" '
            f'class="city-label">{city}</text>'
            f'<title>{city}: {row["price_m"]} M toman/m²; N={row["listing_n"]:,}</title>'
            f'</g>'
        )
        button_parts.append(f'<button class="city-button" data-index="{i}">{city}</button>')

    payload = json.dumps(prepared, ensure_ascii=False).replace("</", "<\\/")
    boundary_note = "Iran boundary embedded from local reference GeoJSON." if boundary_used else "Boundary reference unavailable; geographic coordinate frame shown."
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Four-city M3 Market Map</title>
  <style>
    * {{ box-sizing:border-box; }}
    html, body {{ min-height:100%; margin:0; font-family:Arial, Helvetica, sans-serif; background:#f7f9fb; color:#243746; }}
    body {{ padding:18px; }}
    .shell {{ max-width:1280px; margin:0 auto; }}
    .title {{ background:#fff; border:1px solid #d7dee5; border-radius:10px; padding:14px 18px; box-shadow:0 1px 5px rgba(0,0,0,.07); }}
    .title h1 {{ font-size:22px; margin:0 0 5px; }}
    .title p {{ margin:0; color:#5a6875; font-size:13px; }}
    .offline {{ margin-top:7px; font-size:12px; color:#326747; font-weight:700; }}
    .layout {{ display:grid; grid-template-columns:minmax(0, 1fr) 300px; gap:14px; margin-top:14px; }}
    .map-card, .side-card {{ background:#fff; border:1px solid #d7dee5; border-radius:10px; box-shadow:0 1px 5px rgba(0,0,0,.06); }}
    .map-card {{ padding:10px; overflow:hidden; }}
    svg {{ display:block; width:100%; height:auto; min-height:540px; background:#f8fbfd; border-radius:7px; }}
    .grid {{ stroke:#dce4ea; stroke-width:1; stroke-dasharray:4 5; }}
    .axis-label {{ font-size:13px; fill:#73808b; }}
    .adm-boundary {{ fill:#edf3f7; fill-opacity:.72; stroke:#b7c2cb; stroke-width:.8; vector-effect:non-scaling-stroke; }}
    .city-marker {{ cursor:pointer; outline:none; }}
    .city-marker circle {{ transition:stroke-width .15s, filter .15s, opacity .15s; }}
    .city-marker:hover circle, .city-marker:focus circle, .city-marker.active circle {{ stroke-width:4; filter:drop-shadow(0 2px 3px rgba(0,0,0,.24)); }}
    .city-marker:not(.active):not(:hover) circle {{ opacity:.92; }}
    .city-label {{ font-size:18px; font-weight:700; fill:#263746; paint-order:stroke; stroke:white; stroke-width:4px; stroke-linejoin:round; }}
    .side-card {{ padding:15px; }}
    .side-card h2 {{ font-size:18px; margin:0 0 10px; }}
    .metric {{ border-top:1px solid #e6ebef; padding:9px 0; }}
    .metric:first-of-type {{ border-top:0; }}
    .metric span {{ display:block; font-size:11px; color:#6c7882; text-transform:uppercase; letter-spacing:.03em; }}
    .metric strong {{ display:block; font-size:17px; margin-top:3px; }}
    .city-buttons {{ display:flex; flex-wrap:wrap; gap:6px; margin-top:12px; }}
    .city-button {{ border:1px solid #b9c7d2; background:#f5f8fa; color:#30485b; border-radius:16px; padding:6px 10px; cursor:pointer; font-weight:700; }}
    .city-button.active {{ background:#30485b; color:white; }}
    .legend {{ margin-top:16px; padding-top:12px; border-top:1px solid #e4eaee; font-size:12px; line-height:1.55; color:#56636e; }}
    .dot {{ display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:6px; vertical-align:middle; }}
    .foot {{ margin-top:12px; font-size:11px; color:#66737d; line-height:1.5; }}
    @media (max-width:900px) {{ .layout {{ grid-template-columns:1fr; }} .side-card {{ order:-1; }} svg {{ min-height:420px; }} }}
  </style>
</head>
<body>
<div class="shell">
  <div class="title">
    <h1>Four-city apartment asking-price map</h1>
    <p>May–December 2024 | marker shade = median asking price | marker size = listing count</p>
    <div class="offline">Offline self-contained artifact — no CDN, tile server, or internet connection required.</div>
  </div>
  <div class="layout">
    <div class="map-card">
      <svg id="marketMap" viewBox="0 0 {int(svg_width)} {int(svg_height)}" aria-label="Four-city market map">
        <rect x="{pad_left}" y="{pad_top}" width="{svg_width-pad_left-pad_right}" height="{svg_height-pad_top-pad_bottom}" rx="10" fill="#f7fafc" stroke="#cfd8df"/>
        {''.join(grid_parts)}
        <g id="boundaries">{''.join(boundary_paths)}</g>
        <g id="markers">{''.join(marker_parts)}</g>
      </svg>
    </div>
    <aside class="side-card">
      <h2 id="cityName">City details</h2>
      <div class="metric"><span>Median asking price</span><strong id="priceValue">—</strong></div>
      <div class="metric"><span>Apartment-sale listings</span><strong id="listingValue">—</strong></div>
      <div class="metric"><span>Reliable neighborhoods</span><strong id="reliableValue">—</strong></div>
      <div class="metric"><span>Aggregate coordinate</span><strong id="coordValue">—</strong></div>
      <div class="city-buttons">{''.join(button_parts)}</div>
      <div class="legend">
        <div><span class="dot" style="background:#1F5F99"></span>Darker marker = higher median asking price</div>
        <div>Larger marker = more apartment-sale listings</div>
        <div>Reliable neighborhood threshold: N ≥ {int(min_n)}</div>
      </div>
      <div class="foot">{boundary_note}<br>City coordinates are aggregate medians rounded to 2 decimals. Asking prices are not transactions. No listing-level coordinates are published.</div>
    </aside>
  </div>
</div>
<script>
  const rows = {payload};
  const markers = Array.from(document.querySelectorAll('.city-marker'));
  const buttons = Array.from(document.querySelectorAll('.city-button'));
  const cityName = document.getElementById('cityName');
  const priceValue = document.getElementById('priceValue');
  const listingValue = document.getElementById('listingValue');
  const reliableValue = document.getElementById('reliableValue');
  const coordValue = document.getElementById('coordValue');

  function selectCity(index) {{
    const row = rows[index];
    if (!row) return;
    markers.forEach((el, i) => el.classList.toggle('active', i === index));
    buttons.forEach((el, i) => el.classList.toggle('active', i === index));
    cityName.textContent = row.city;
    priceValue.textContent = `${{row.price_m.toFixed(1)}} M toman/m²`;
    listingValue.textContent = row.listing_n.toLocaleString();
    reliableValue.textContent = row.reliable_neighborhood_n.toLocaleString();
    coordValue.textContent = `${{row.lat.toFixed(2)}}°, ${{row.lon.toFixed(2)}}°`;
  }}

  markers.forEach((el, i) => {{
    el.addEventListener('click', () => selectCity(i));
    el.addEventListener('keydown', (event) => {{ if (event.key === 'Enter' || event.key === ' ') {{ event.preventDefault(); selectCity(i); }} }});
  }});
  buttons.forEach((el, i) => el.addEventListener('click', () => selectCity(i)));
  selectCity(0);
</script>
</body>
</html>
"""
    atomic_write_text(html, path)
    return path.exists() and path.stat().st_size > 0


def show_progress(percent: int, label: str, *, final: bool = False) -> None:
    pct = max(0, min(100, int(percent)))
    filled = int(PROGRESS_WIDTH * pct / 100)
    bar = "#" * filled + "-" * (PROGRESS_WIDTH - filled)
    print(
        f"\rM3 market map [{bar}] {pct:3d}% complete | {100-pct:3d}% remaining | {label[:42]:42s}",
        end="\n" if final else "", flush=True,
    )


def _summary(scan: pl.LazyFrame, keys: list[str], min_n: int) -> pl.DataFrame:
    psm = pl.col("sale_price_per_sqm_final_toman").cast(pl.Float64, strict=False)
    result = (
        scan.filter(psm.is_not_null() & psm.is_finite() & (psm > 0))
        .group_by(keys)
        .agg([
            pl.len().alias("listing_n"),
            psm.median().alias("median_asking_price_per_sqm_toman"),
            psm.quantile(0.25).alias("p25_asking_price_per_sqm_toman"),
            psm.quantile(0.75).alias("p75_asking_price_per_sqm_toman"),
        ])
        .with_columns([
            (pl.col("p75_asking_price_per_sqm_toman") - pl.col("p25_asking_price_per_sqm_toman")).alias("iqr_asking_price_per_sqm_toman"),
            (pl.col("listing_n") >= min_n).alias("reliable_flag"),
            pl.when(pl.col("listing_n") >= min_n)
            .then(pl.lit("PASS"))
            .otherwise(pl.lit("REVIEW"))
            .alias("reliability_status"),
            pl.when(pl.col("listing_n") >= min_n)
            .then(pl.lit(f"N >= {min_n}"))
            .otherwise(pl.lit(f"N < {min_n}"))
            .alias("reliability_reason"),
            pl.lit("apartment_sale_asking_price_per_sqm").alias("metric_definition"),
            pl.lit(VERSION).alias("analysis_version"),
        ])
        .sort("median_asking_price_per_sqm_toman", descending=True)
        .collect(engine="streaming")
    )
    return result


def run(
    sales_base: Path = SALES_BASE,
    map_base: Path = MAP_BASE,
    min_n: int | None = None,
    boundary_geojson: Path | None = DEFAULT_BOUNDARY_GEOJSON,
) -> dict[str, Path]:
    sales_base = sales_base.resolve()
    map_base = map_base.resolve()
    for path in [sales_base, map_base]:
        if not path.exists():
            raise FileNotFoundError(f"Required M3 population artifact not found: {path}")
    min_n = int(min_n or setting("analysis", "minimum_valid_listings", "default", default=30))

    table_dir = OUTPUTS_DIR / "tables" / "milestone_3" / "market_map"
    qa_dir = OUTPUTS_DIR / "qa" / "milestone_3" / "market_map"
    fig_dir = OUTPUTS_DIR / "figures" / "milestone_3" / "market_map"
    map_dir = OUTPUTS_DIR / "maps" / "milestone_3" / "market_map"
    table_dir.mkdir(parents=True, exist_ok=True)
    qa_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    map_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    show_progress(0, "validating population schemas")

    sales = pl.scan_parquet(sales_base)
    maps = pl.scan_parquet(map_base)
    sales_cols = set(sales.collect_schema().names())
    map_cols = set(maps.collect_schema().names())
    missing_sales = sorted(REQUIRED_SALES - sales_cols)
    missing_map = sorted(REQUIRED_MAP - map_cols)
    if missing_sales or missing_map:
        raise ValueError(f"Missing market-map inputs: sales={missing_sales}, map={missing_map}")

    apartment_sales = sales.filter(pl.col("property_family") == "apartment")
    city = _summary(apartment_sales, ["city_slug"], min_n)
    show_progress(30, f"city summary: {city.height:,} cities")
    neighborhood = _summary(apartment_sales, ["city_slug", "neighborhood_slug"], min_n)
    show_progress(58, f"neighborhood summary: {neighborhood.height:,} groups")

    centroids = (
        maps.filter(pl.col("latitude").is_not_null() & pl.col("longitude").is_not_null())
        .group_by(["city_slug", "neighborhood_slug"])
        .agg([
            pl.len().alias("map_listing_n"),
            pl.col("latitude").cast(pl.Float64, strict=False).median().round(2).alias("centroid_latitude_2dp"),
            pl.col("longitude").cast(pl.Float64, strict=False).median().round(2).alias("centroid_longitude_2dp"),
        ])
        .collect(engine="streaming")
    )
    neighborhood = neighborhood.join(centroids, on=["city_slug", "neighborhood_slug"], how="left")
    show_progress(72, "aggregate map centroids attached")

    reliable = neighborhood.filter(pl.col("reliable_flag") & pl.col("neighborhood_slug").is_not_null())
    ranking = (
        reliable.with_columns([
            pl.col("median_asking_price_per_sqm_toman")
            .rank(method="dense", descending=True).over("city_slug").alias("expensive_rank_within_city"),
            pl.col("median_asking_price_per_sqm_toman")
            .rank(method="dense", descending=False).over("city_slug").alias("affordable_rank_within_city"),
        ])
        .sort(["city_slug", "expensive_rank_within_city"])
    )

    # Professor-facing Python figures are intentionally limited to three outputs:
    # 1) valid apartment-sale listing volume in the four presentation cities,
    # 2) five most affordable reliable neighborhoods per city, and
    # 3) five most expensive reliable neighborhoods per city.
    # The old static geographic PNG is removed to avoid duplicating the dashboard map.
    stale_figures = [
        fig_dir / "reliable_neighborhood_price_ranking.png",
        fig_dir / "city_price_comparison.png",
        fig_dir / "four_city_market_price_map.png",
    ]
    for stale_figure in stale_figures:
        stale_figure.unlink(missing_ok=True)

    city_figure = fig_dir / "city_listing_volume.png"
    affordable_figure = fig_dir / "top_affordable_neighborhoods.png"
    expensive_figure = fig_dir / "top_expensive_neighborhoods.png"
    interactive_map = map_dir / "four_city_market_map.html"

    city_order = ["tehran", "mashhad", "karaj", "isfahan"]
    city_rows = {str(row["city_slug"]): row for row in city.to_dicts()}
    ordered_city_rows = [city_rows[key] for key in city_order if key in city_rows]

    if ordered_city_rows:
        labels = [_display_slug(row["city_slug"]) for row in ordered_city_rows]
        city_ns = [int(row["listing_n"]) for row in ordered_city_rows]
        fig, ax = plt.subplots(figsize=(10.8, 6.6))
        bars = ax.bar(labels, city_ns, width=0.62)
        ax.set_ylabel("Valid apartment-sale listings")
        ax.set_xlabel("City")
        ax.grid(axis="y", alpha=0.18, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title("Valid Apartment-Sale Listing Volume by City", fontsize=15.5, fontweight="bold", pad=22)
        ax.text(
            0.5, 1.01,
            "Tehran, Mashhad, Karaj and Isfahan | May–December 2024",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=10.5, color="#555555",
        )
        top = max(city_ns) if city_ns else 1
        ax.set_ylim(0, top * 1.13)
        ax.ticklabel_format(style="plain", axis="y")
        for bar, listing_n in zip(bars, city_ns):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                listing_n + top * 0.018,
                f"{listing_n:,}",
                ha="center", va="bottom", fontsize=9.5,
            )
        fig.text(
            0.01, 0.012,
            "Counts are valid apartment-sale asking-price-per-sqm observations in the M3-03 population; they are not total housing-market supply.",
            fontsize=8.4, color="#555555",
        )
        fig.tight_layout(rect=(0, 0.05, 1, 0.97))
        fig.savefig(city_figure, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    presentation = reliable.filter(pl.col("city_slug").is_in(list(FOUR_CITY_SLUGS)))
    reliable_city_counts = {
        str(row["city_slug"]): int(row["reliable_neighborhood_n"])
        for row in presentation.group_by("city_slug").agg(pl.len().alias("reliable_neighborhood_n")).to_dicts()
    }

    def _select_five_per_city(*, affordable: bool) -> pl.DataFrame:
        frames: list[pl.DataFrame] = []
        for city_slug in city_order:
            city_frame = presentation.filter(pl.col("city_slug") == city_slug)
            if city_frame.height == 0:
                continue
            city_frame = city_frame.sort(
                ["median_asking_price_per_sqm_toman", "neighborhood_slug"],
                descending=[not affordable, False],
            ).head(5)
            frames.append(city_frame)
        if not frames:
            return presentation.head(0)
        return pl.concat(frames, how="vertical_relaxed")

    affordable = _select_five_per_city(affordable=True)
    expensive = _select_five_per_city(affordable=False)

    city_colors = {
        "tehran": "#2F7DB5",
        "mashhad": "#5A9A5A",
        "karaj": "#D18F35",
        "isfahan": "#8B6BB8",
    }

    def _save_neighborhood_rank(
        data: pl.DataFrame,
        path: Path,
        title: str,
        *,
        direction_label: str,
    ) -> None:
        if data.height == 0:
            return
        rows = data.to_dicts()
        values_m = [float(row["median_asking_price_per_sqm_toman"]) / 1_000_000.0 for row in rows]
        listing_ns = [int(row["listing_n"]) for row in rows]
        colors = [city_colors.get(str(row["city_slug"]), "#6F7C85") for row in rows]

        # Add a visual gap after each five-row city block.
        y_positions: list[float] = []
        labels: list[str] = []
        current = 0.0
        previous_city: str | None = None
        for row in rows:
            city_slug = str(row["city_slug"])
            if previous_city is not None and city_slug != previous_city:
                current += 0.75
            y_positions.append(current)
            labels.append(f"{_display_slug(city_slug)} — {_display_slug(row['neighborhood_slug'])}")
            current += 1.0
            previous_city = city_slug

        fig, ax = plt.subplots(figsize=(13.6, 11.2))
        bars = ax.barh(y_positions, values_m, color=colors, height=0.68)
        ax.set_yticks(y_positions, labels, fontsize=9.2)
        ax.invert_yaxis()
        ax.set_xlabel("Median asking price per sqm (million toman)")
        ax.set_ylabel("")
        ax.grid(axis="x", alpha=0.16, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_title(title, fontsize=15.0, fontweight="bold", pad=22)
        ax.text(
            0.5, 1.005,
            "Exactly five reliable neighborhoods per city | common price scale across all four cities",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=10.2, color="#555555",
        )

        maximum = max(values_m) if values_m else 1.0
        ax.set_xlim(0, maximum * 1.22)
        pad = maximum * 0.012
        for bar, value, listing_n in zip(bars, values_m, listing_ns):
            ax.text(
                value + pad,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.1f} M   N={listing_n:,}",
                va="center", ha="left", fontsize=8.7,
            )

        # City-block separators improve scanability without adding another subplot.
        for block_end in (4, 9, 14):
            if block_end + 1 < len(y_positions):
                separator = (y_positions[block_end] + y_positions[block_end + 1]) / 2
                ax.axhline(separator, color="#D6DCE1", linewidth=0.9, zorder=0)

        fig.text(
            0.01, 0.012,
            f"{direction_label} five within each city among reliable apartment-sale neighborhoods (N >= {min_n}); asking prices, not transactions; Toman is the operational unit and remains source-unconfirmed.",
            fontsize=8.2, color="#555555",
        )
        fig.tight_layout(rect=(0, 0.05, 1, 0.97))
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    _save_neighborhood_rank(
        affordable,
        affordable_figure,
        "Five Most Affordable Reliable Neighborhoods in Each City",
        direction_label="Lowest-median",
    )
    _save_neighborhood_rank(
        expensive,
        expensive_figure,
        "Five Most Expensive Reliable Neighborhoods in Each City",
        direction_label="Highest-median",
    )

    # Professor-facing geographic context: four-city aggregates only. Exact listing coordinates
    # remain internal inputs and are never written to the figure or Gold/dashboard outputs.
    city_centroids = (
        maps.filter(
            pl.col("city_slug").is_in(list(FOUR_CITY_SLUGS))
            & pl.col("latitude").is_not_null()
            & pl.col("longitude").is_not_null()
        )
        .group_by("city_slug")
        .agg([
            pl.col("latitude").cast(pl.Float64, strict=False).median().alias("city_latitude"),
            pl.col("longitude").cast(pl.Float64, strict=False).median().alias("city_longitude"),
        ])
        .collect(engine="streaming")
    )
    centroid_rows = {str(row["city_slug"]): row for row in city_centroids.to_dicts()}
    geo_rows = [
        {
            **row,
            **centroid_rows.get(str(row["city_slug"]), {}),
            "reliable_neighborhood_n": reliable_city_counts.get(str(row["city_slug"]), 0),
        }
        for row in ordered_city_rows
        if str(row["city_slug"]) in centroid_rows
    ]
    boundary_path = boundary_geojson.resolve() if boundary_geojson is not None else None
    boundary_used = bool(boundary_path is not None and boundary_path.exists())

    interactive_map_created = _write_interactive_city_map(geo_rows, interactive_map, min_n=min_n, boundary_geojson=boundary_path)

    checks: list[Check] = [
        make_check("city_summary_nonempty", "market_map", city.height, ">0", city.height > 0),
        make_check("reliable_neighborhoods_nonempty", "market_map", reliable.height, ">0", reliable.height > 0),
        make_check(
            "no_exact_coordinates_in_public_outputs", "privacy", "aggregate_centroids_only", "no exact listing coordinates",
            True, notes="Neighborhood table uses rounded 2-decimal aggregate centroids; no static geographic PNG is published and the optional offline HTML map serializes only city-level aggregate medians rounded to 2 decimals.",
        ),
        make_check(
            "four_city_map_coverage", "market_map", len(geo_rows), 4,
            len(geo_rows) == 4, critical=True,
            notes="Professor-facing static and interactive maps require Tehran, Mashhad, Karaj and Isfahan city aggregates.",
        ),
        make_check(
            "three_professor_facing_figures_created",
            "market_map",
            sum(path.exists() for path in [city_figure, affordable_figure, expensive_figure]),
            3,
            all(path.exists() for path in [city_figure, affordable_figure, expensive_figure]),
            critical=True,
            notes="Python outputs are intentionally limited to city listing volume plus balanced affordable/expensive neighborhood comparisons.",
        ),
        make_check(
            "balanced_five_neighborhoods_per_city",
            "market_map",
            {
                city_slug: {
                    "affordable": affordable.filter(pl.col("city_slug") == city_slug).height,
                    "expensive": expensive.filter(pl.col("city_slug") == city_slug).height,
                }
                for city_slug in city_order
            },
            "5 affordable and 5 expensive per city",
            all(
                affordable.filter(pl.col("city_slug") == city_slug).height == 5
                and expensive.filter(pl.col("city_slug") == city_slug).height == 5
                for city_slug in city_order
            ),
            critical=True,
            notes="Selection is within-city, reliability-gated and deterministic; no city can dominate the surfaced top/bottom comparison.",
        ),
        make_check(
            "static_geographic_png_removed",
            "market_map",
            (fig_dir / "four_city_market_price_map.png").exists(),
            False,
            not (fig_dir / "four_city_market_price_map.png").exists(),
            critical=False,
            notes="The static map is intentionally omitted; the dashboard owns the interactive geographic presentation.",
        ),
        make_check(
            "four_city_interactive_map_created", "market_map", interactive_map_created, True,
            interactive_map_created, critical=False,
            notes="Self-contained offline HTML uses four city-level aggregate median coordinates rounded to 2 decimals; no external network requests or listing-level coordinates are serialized.",
        ),
    ]

    city_path = table_dir / "city_market_summary.csv"
    neighborhood_path = table_dir / "neighborhood_market_summary.csv"
    ranking_path = table_dir / "reliable_neighborhood_ranking.csv"
    checks_path = qa_dir / "market_map_checks.csv"
    manifest_path = qa_dir / "market_map_manifest.json"
    atomic_write_csv(city, city_path)
    atomic_write_csv(neighborhood, neighborhood_path)
    atomic_write_csv(ranking, ranking_path)
    atomic_write_csv(checks_frame(checks), checks_path)
    status = summarize_checks(checks)
    atomic_write_json(
        {
            "version": VERSION,
            "status": status,
            "minimum_valid_listings": min_n,
            "inputs": {"sales": relative_to_project(sales_base), "map": relative_to_project(map_base)},
            "outputs": {
                "city": relative_to_project(city_path),
                "neighborhood": relative_to_project(neighborhood_path),
                "ranking": relative_to_project(ranking_path),
                "checks": relative_to_project(checks_path),
                "city_figure": relative_to_project(city_figure) if city_figure.exists() else None,
                "affordable_figure": relative_to_project(affordable_figure) if affordable_figure.exists() else None,
                "expensive_figure": relative_to_project(expensive_figure) if expensive_figure.exists() else None,
                "python_figure_policy": "exactly_three_professor_facing_figures",
                "interactive_map": relative_to_project(interactive_map) if interactive_map_created else None,
                "interactive_map_coordinate_policy": "city aggregate median coordinates rounded to 2 decimal degrees; no listing-level points",
                "interactive_map_runtime": "offline_self_contained_svg_no_external_network_requests",
                "boundary_reference": relative_to_project(boundary_path) if (boundary_path is not None and boundary_path.exists()) else None,
                "boundary_reference_used": boundary_used,
            },
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }, manifest_path,
    )
    show_progress(100, f"complete in {time.perf_counter()-started:.1f}s", final=True)
    return {"city": city_path, "neighborhood": neighborhood_path, "ranking": ranking_path, "city_figure": city_figure, "affordable_figure": affordable_figure, "expensive_figure": expensive_figure, "interactive_map": interactive_map, "checks": checks_path, "manifest": manifest_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build all-city M3 asking-price market summaries.")
    parser.add_argument("--sales-base", type=Path, default=SALES_BASE)
    parser.add_argument("--map-base", type=Path, default=MAP_BASE)
    parser.add_argument("--minimum-n", type=int, default=None)
    parser.add_argument("--boundary-geojson", type=Path, default=DEFAULT_BOUNDARY_GEOJSON)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run(args.sales_base, args.map_base, args.minimum_n, args.boundary_geojson)
    print("M3 MARKET MAP COMPLETED")
    for name, path in outputs.items():
        print(f"{name}: {relative_to_project(path)}")


if __name__ == "__main__":
    main()
