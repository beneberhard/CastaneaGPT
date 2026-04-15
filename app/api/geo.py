# app/api/geo.py

# app/api/geo.py
from fastapi import APIRouter, Query, HTTPException
from sqlalchemy import text
from app.api.db import get_engine
import json

router = APIRouter(prefix="/geo", tags=["geo"])

@router.get("/stands")
def get_stands(
    bbox: str | None = Query(
        None,
        description="minLon,minLat,maxLon,maxLat (EPSG:4326). If omitted, uses a default demo bbox."
    )
):
    # Default bbox (Italy-ish). Adjust as you like.
    if not bbox:
        minx, miny, maxx, maxy = 6.0, 36.0, 19.0, 47.5
    else:
        minx, miny, maxx, maxy = map(float, bbox.split(","))

    sql = text("""
        SELECT id, name, ST_AsGeoJSON(geom) AS geojson
        FROM stands
        WHERE geom && ST_MakeEnvelope(:minx, :miny, :maxx, :maxy, 4326)
        LIMIT 200;
    """)

    with get_engine().connect() as conn:
        rows = conn.execute(
            sql, {"minx": minx, "miny": miny, "maxx": maxx, "maxy": maxy}
        ).fetchall()

    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"id": r.id, "name": r.name},
                "geometry": json.loads(r.geojson) if r.geojson else None,
            }
            for r in rows
            if r.geojson
        ],
    }

@router.get("/stands/{stand_id}")
def get_stand_by_id(stand_id: int):
    sql = text("""
        SELECT id, name, ST_AsGeoJSON(geom) AS geojson
        FROM stands
        WHERE id = :id
        LIMIT 1;
    """)

    with get_engine().connect() as conn:
        row = conn.execute(sql, {"id": stand_id}).fetchone()

    if not row or not row.geojson:
        raise HTTPException(status_code=404, detail="Stand not found")

    return {
        "type": "Feature",
        "properties": {"id": row.id, "name": row.name},
        "geometry": json.loads(row.geojson),
    }

@router.get("/stands/{stand_id}/summary")
def get_stand_summary(stand_id: int):
    sql = text("""
        SELECT
            id,
            name,
            species,
            management_type,
            age_class,
            altitude_m,
            notes,
            ST_Area(geom::geography) / 10000.0 AS area_ha,
            ST_Perimeter(geom::geography) AS perimeter_m
        FROM stands
        WHERE id = :id
        LIMIT 1;
    """)

    with get_engine().connect() as conn:
        row = conn.execute(sql, {"id": stand_id}).fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Stand not found")

    return {
        "id": row.id,
        "name": row.name,
        "species": row.species,
        "management_type": row.management_type,
        "age_class": row.age_class,
        "altitude_m": row.altitude_m,
        "area_ha": round(row.area_ha, 3) if row.area_ha else None,
        "perimeter_m": round(row.perimeter_m, 1) if row.perimeter_m else None,
        "notes": row.notes,
    }
