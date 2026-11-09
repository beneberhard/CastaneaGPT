"""Pl@ntNet disease adapter. Call synchronous identify_images in a worker thread.

Inputs must already have passed validation.validate_images. No raw responses,
request URLs, credentials, or exception text are logged or returned.
"""

import math
import os

import requests

from app.identification.schemas import DiseaseCandidate, IdentificationResult


DISEASE_ENDPOINT = "https://my-api.plantnet.org/v2/diseases/identify"
TIMEOUT = (5, 30)  # Connection and read-inactivity timeouts, in seconds.
# Prototype heuristic, not a calibrated diagnostic probability threshold.
LOW_CONFIDENCE_THRESHOLD = 0.5
ORGAN_MAP = {
    "auto": "auto", "leaf": "leaf", "flower": "flower",
    "fruit": "fruit", "bark": "bark", "branch": "auto",
    "whole_tree": "auto",
}


def normalize_response(payload: object) -> IdentificationResult:
    """Normalize the documented results[].name/score/description structure."""
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        return IdentificationResult(
            status="api_error", message="Pl@ntNet returned an unexpected response."
        )

    candidates = []
    for item in payload["results"]:
        if not isinstance(item, dict):
            continue
        code = item.get("name")
        if not isinstance(code, str) or not code.strip():
            continue
        score = item.get("score")
        if score is not None and (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(score)
            or not 0 <= score <= 1
        ):
            continue
        description = item.get("description")
        candidates.append(DiseaseCandidate(
            eppo_code=code.strip(),
            description=description if isinstance(description, str) else None,
            # The documented API has no separate scientific-name field.
            # Preserve description rather than guessing a name from free text.
            scientific_name=None,
            confidence=score,
        ))

    version = payload.get("version")
    version = version if isinstance(version, str) else None
    if not candidates:
        malformed = bool(payload["results"])
        return IdentificationResult(
            status="api_error" if malformed else "no_result",
            message=("Pl@ntNet returned no usable candidate data." if malformed else
                     "No disease candidate was returned. Evidence is insufficient; "
                     "the species or condition may be outside Pl@ntNet's coverage."),
            model_version=version,
        )

    candidates.sort(
        key=lambda candidate: candidate.confidence if candidate.confidence is not None else -1,
        reverse=True,
    )
    best = candidates[0].confidence
    low = best is None or best < LOW_CONFIDENCE_THRESHOLD
    return IdentificationResult(
        status="low_confidence" if low else "ok",
        candidates=candidates[:5],
        message=(
            "Identification is uncertain; low or unavailable confidence. "
            "An unknown cause remains possible."
            if low else
            "Preliminary image candidates only, not a definitive phytosanitary diagnosis. "
            "Host relevance must be checked against corpus evidence."
        ),
        model_version=version,
    )


def identify_images(
    images: list[tuple[bytes, str]],
    organ: str = "auto",
    language: str = "en",
) -> IdentificationResult:
    """Identify validated images of one plant using the server-side key.

    No retries: a request may consume quota even if the response times out.
    Empty results do not prove an unsupported species; keep that uncertainty.
    """
    if organ not in ORGAN_MAP:
        raise ValueError("Unsupported plant organ.")
    if not 1 <= len(images) <= 5:
        raise ValueError("Provide between one and five validated images.")
    if any(mime not in {"image/jpeg", "image/png"} for _, mime in images):
        raise ValueError("Only validated JPEG and PNG images are accepted.")

    api_key = os.getenv("PLANTNET_API_KEY", "").strip()
    if not api_key:
        return IdentificationResult(
            status="not_configured", message="Image identification is not configured on the server."
        )

    files = [
        ("images", (f"image_{i}.{'jpg' if mime == 'image/jpeg' else 'png'}", content, mime))
        for i, (content, mime) in enumerate(images, start=1)
    ]
    try:
        with requests.post(
            DISEASE_ENDPOINT,
            params={
                "api-key": api_key,
                "lang": language,
                "nb-results": 5,
                "include-related-images": "false",
                "no-reject": "false",
            },
            files=files,
            data=[("organs", ORGAN_MAP[organ]) for _ in images],
            timeout=TIMEOUT,
            allow_redirects=False,
        ) as response:
            if response.status_code != 200:
                messages = {
                    400: "Pl@ntNet could not accept these images or parameters.",
                    401: "Pl@ntNet authentication failed. Check the server configuration.",
                    403: "Pl@ntNet access was denied. Check the server account permissions.",
                    404: "Pl@ntNet returned no identification or the service route is unavailable.",
                    413: "Pl@ntNet rejected the upload size.",
                    422: "Pl@ntNet could not process these images. Try clearer photographs.",
                    429: "Pl@ntNet's request limit was reached. Please try again later.",
                }
                return IdentificationResult(
                    status="api_error",
                    message=messages.get(response.status_code,
                                         "Pl@ntNet is unavailable. Please try again later."),
                )
            return normalize_response(response.json())
    except requests.Timeout:
        return IdentificationResult(
            status="timeout", message="Pl@ntNet timed out. Please try again later."
        )
    except (requests.RequestException, ValueError):
        return IdentificationResult(
            status="api_error", message="Pl@ntNet could not return a usable response. Please try again later."
        )
