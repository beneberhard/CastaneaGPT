from fastapi import APIRouter, HTTPException

from app.utils.stand_context import compute_ndvi_stats_for_stand

router = APIRouter()


@router.get("/stands/{stand_id}/ndvi")
def stand_ndvi(stand_id: int):
    stats = compute_ndvi_stats_for_stand(stand_id)

    if not stats:
        raise HTTPException(status_code=404, detail="No NDVI data found for stand")

    series = []
    for (year, month), values in stats.items():
        series.append({
            "year": year,
            "month": month,
            "label": f"{year}-{month:02d}",
            "mean": values["mean"],
            "min": values["min"],
            "max": values["max"],
            "std": values["std"],
            "range": values["range"],
        })

    return {
        "latest": series[-1],
        "series": series,
    }
