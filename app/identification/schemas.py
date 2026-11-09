from typing import Literal

from pydantic import BaseModel, Field


class DiseaseCandidate(BaseModel):
    eppo_code: str
    description: str | None = None
    scientific_name: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)


class IdentificationResult(BaseModel):
    provider: str = "plantnet"
    status: Literal[
        "ok",
        "low_confidence",
        "no_result",
        "unsupported",
        "not_configured",
        "timeout",
        "api_error",
    ]
    candidates: list[DiseaseCandidate] = Field(default_factory=list)
    message: str
    model_version: str | None = None