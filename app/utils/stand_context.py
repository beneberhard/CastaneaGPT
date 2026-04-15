import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.mask import mask

NDVI_DIR = Path("data/NDVI_rasters")


def compute_ndvi_stats(polygon_geojson):

    results = {}

    for tif in NDVI_DIR.glob("ndvi_*.tif"):

        parts = tif.stem.split("_")
        month = int(parts[1])
        year = int(parts[2])

        with rasterio.open(tif) as src:

            out_img, _ = mask(src, polygon_geojson, crop=True)

            values = out_img[0]
            values = values[values != src.nodata]

            if len(values) == 0:
                continue

            stats = {
                "mean": float(np.mean(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "std": float(np.std(values)),
                "range": float(np.max(values) - np.min(values))
            }

            results[(year, month)] = stats

    return dict(sorted(results.items()))


def compute_ndvi_stats_for_stand(stand_id):
    """Return NDVI stats for a stand keyed by (year, month), or an empty dict."""
    from sqlalchemy import text

    from app.api.db import get_engine

    sql = text("""
        SELECT ST_AsGeoJSON(geom)
        FROM stands
        WHERE id = :id
        LIMIT 1;
    """)

    with get_engine().connect() as conn:
        row = conn.execute(sql, {"id": stand_id}).fetchone()

    if not row:
        return {}

    polygon = [json.loads(row[0])]
    return compute_ndvi_stats(polygon)


def compare_ndvi(stats, year1, month1, year2, month2):

    key1 = (year1, month1)
    key2 = (year2, month2)

    if key1 not in stats or key2 not in stats:
        return None

    s1 = stats[key1]
    s2 = stats[key2]

    return {
        "mean_change": s2["mean"] - s1["mean"],
        "max_change": s2["max"] - s1["max"],
        "range_change": s2["range"] - s1["range"]
    }


def format_ndvi_stats(stats):

    lines = []

    for (year, month), s in stats.items():

        lines.append(
            f"{year}-{month:02d}: "
            f"mean={s['mean']:.3f}, "
            f"min={s['min']:.3f}, "
            f"max={s['max']:.3f}, "
            f"range={s['range']:.3f}"
        )

    return "\n".join(lines)


def build_stand_context_block(stand_id):
    """Return stand metadata for prompt injection, or an empty string."""
    from sqlalchemy import text

    from app.api.db import get_engine

    sql = text("""
        SELECT
            name,
            species,
            management_type,
            age_class,
            altitude_m,
            ST_Area(geom::geography)/10000.0 AS area_ha,
            notes
        FROM stands
        WHERE id = :id
        LIMIT 1;
    """)

    with get_engine().connect() as conn:
        row = conn.execute(sql, {"id": stand_id}).fetchone()

    if not row:
        return ""

    return f"""
Stand context:
- Name: {row.name}
- Species: {row.species}
- Management: {row.management_type}
- Age class: {row.age_class}
- Area (ha): {round(row.area_ha, 3) if row.area_ha else "unknown"}
- Altitude (m): {row.altitude_m}
- Notes: {row.notes}
"""


def build_ndvi_context_block(stand_id):
    """Return an NDVI context block for prompt injection, or an empty string."""
    try:
        ndvi_stats = compute_ndvi_stats_for_stand(stand_id)
        ndvi_text = format_ndvi_stats(ndvi_stats)

        if not ndvi_text:
            return ""

        return f"""

NDVI statistics for this stand (year-month):
{ndvi_text}

Interpretation hint:
Higher NDVI values indicate denser or healthier vegetation.
When asked about vegetation dynamics, compare NDVI values across years.
"""
    except Exception as e:
        print("NDVI calculation failed:", e)
        return ""
