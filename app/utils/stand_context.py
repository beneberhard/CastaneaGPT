import json
import re
from pathlib import Path

import numpy as np
import rasterio
from rasterio.mask import mask

NDVI_DIR = Path("data/NDVI_rasters")

NDVI_QUERY_PATTERNS = (
    # English
    r"\bndvi\b",
    r"\bvegetation\b",
    r"\bvegetative\b",
    r"\bgreenness\b",
    r"\bcanopy\b",
    r"\bvigor\b",
    r"\bvigour\b",
    r"\bbiomass\b",
    r"\bhealth(?:ier)?\b",
    r"\bstress(?:ed)?\b",
    r"\bremote\s+sensing\b",
    r"\bsatellite\b",
    r"\bimagery\b",
    r"\btime\s+series\b",
    r"\btrend\b",
    r"\bchange(?:s|d)?\b",
    r"\bdynamics?\b",
    r"\bcompare\b",
    r"\bmonitor(?:ing)?\b",
    r"\bseason(?:al)?\b",
    # Italian
    r"\bvegetazione\b",
    r"\bchioma\b",
    r"\bbiomassa\b",
    r"\bsalute\s+della\s+vegetazione\b",
    r"\bstress\s+vegetativo\b",
    r"\btelerilevamento\b",
    r"\bsatellit(?:e|are)\b",
    r"\bimmagin(?:e|i)\b",
    r"\bserie\s+temporale\b",
    r"\btendenz[ae]\b",
    r"\bcambiament(?:o|i)\b",
    r"\bdinamic(?:a|he)\b",
    r"\bconfronta(?:re|to)?\b",
    r"\bmonitoragg(?:io|i)\b",
    r"\bstagional(?:e|i)\b",
    # German
    r"\bvegetation\b",
    r"\bvegetationszustand\b",
    r"\bkronendach\b",
    r"\bbiomasse\b",
    r"\bgesundheit\s+der\s+vegetation\b",
    r"\bvegetationsstress\b",
    r"\bfernerkundung\b",
    r"\bsatellit(?:en|enbild|enbilder)?\b",
    r"\bzeitreihe[n]?\b",
    r"\btrend(?:s)?\b",
    r"\bveranderung(?:en)?\b",
    r"\bveränderung(?:en)?\b",
    r"\bdynamik\b",
    r"\bvergleichen?\b",
    r"\bmonitoring\b",
    r"\bsaisonal(?:e|er|en)\b",
)


def should_include_ndvi_context(question: str) -> bool:
    """Return True when the user question is explicitly asking for NDVI-like analysis."""
    if not question:
        return False

    normalized = question.lower()
    return any(re.search(pattern, normalized) for pattern in NDVI_QUERY_PATTERNS)


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
