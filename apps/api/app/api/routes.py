"""HTTP endpoints."""

from __future__ import annotations

import time
from typing import Annotated

from fastapi import APIRouter, File, Request, UploadFile

from app.schemas.prediction import ErrorResponse, HealthResponse, PredictResponse
from app.services.images import ApiError, decode_image
from app.settings import Settings

router = APIRouter()

_ERRORS = {code: {"model": ErrorResponse} for code in (400, 413, 415, 503)}


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    predictor = getattr(request.app.state, "predictor", None)
    return HealthResponse(
        status="ok",
        model_loaded=predictor is not None,
        model_name=predictor.meta.get("name") if predictor else None,
    )


@router.post("/predict", response_model=PredictResponse, responses=_ERRORS)
def predict(request: Request, file: Annotated[UploadFile, File()]) -> PredictResponse:
    # Sync handler: FastAPI runs it in a worker thread, so ONNX Runtime doesn't block the loop.
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise ApiError(503, "model_not_loaded", "The model bundle is not loaded.")
    settings: Settings = request.app.state.settings
    max_bytes = int(settings.max_upload_mb * 1e6)
    data = file.file.read(max_bytes + 1)  # never read more than the limit + 1 byte
    img = decode_image(data, max_bytes, settings.max_pixels)

    t0 = time.perf_counter()
    pred = predictor.predict(img)
    return PredictResponse(
        label=pred.label,
        confidence=pred.confidence,
        probabilities=pred.probabilities,
        model_name=predictor.meta.get("name", ""),
        latency_ms=round((time.perf_counter() - t0) * 1000, 2),
    )
