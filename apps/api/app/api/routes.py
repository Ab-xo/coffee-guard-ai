"""HTTP endpoints."""

from __future__ import annotations

import logging
import time
from typing import Annotated

from fastapi import APIRouter, File, Request, UploadFile

from app.schemas.prediction import (
    AnalyzeResponse,
    ErrorResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictResponse,
)
from app.services.images import ApiError, decode_image
from app.settings import Settings

router = APIRouter()
log = logging.getLogger("coffeeguard.api")

_ERRORS = {code: {"model": ErrorResponse} for code in (400, 413, 415, 422, 503)}


def _pipeline(request: Request):
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise ApiError(503, "model_not_loaded", "The model bundle is not loaded.")
    return pipeline


def _run(request: Request, file: UploadFile, with_cam: bool):
    # Sync handlers: FastAPI runs them in a worker thread, so ONNX Runtime doesn't block
    # the event loop.
    pipeline = _pipeline(request)
    settings: Settings = request.app.state.settings
    max_bytes = int(settings.max_upload_mb * 1e6)
    data = file.file.read(max_bytes + 1)  # never read more than the limit + 1 byte
    img = decode_image(data, max_bytes, settings.max_pixels)
    t0 = time.perf_counter()
    res = pipeline.run(img, with_cam=with_cam)
    latency = round((time.perf_counter() - t0) * 1000, 2)
    d = res.decision
    request.state.log_extra = {"status": d.status, "reason": d.reason, "label": d.label}
    base = {
        "status": d.status,
        "reason": d.reason,
        "label": d.label,
        "confidence": d.confidence,
        "prediction_set": d.prediction_set,
        "issues": d.issues,
        "advice": d.advice,
        "model_version": pipeline.version,
        "latency_ms": latency,
    }
    return pipeline, res, base


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    pipeline = getattr(request.app.state, "pipeline", None)
    return HealthResponse(
        status="ok",
        model_loaded=pipeline is not None,
        model_name=pipeline.meta.get("name") if pipeline else None,
        model_version=pipeline.version if pipeline else None,
    )


@router.get("/model-info", response_model=ModelInfoResponse, responses=_ERRORS)
def model_info(request: Request) -> ModelInfoResponse:
    p = _pipeline(request)
    m = p.meta
    return ModelInfoResponse(
        name=m.get("name", ""),
        version=p.version,
        architecture=m.get("model", ""),
        classes=p.classes,
        image_size=m["img_size"],
        data_fingerprint=m.get("data_fingerprint"),
        source_run=m.get("source_run"),
        metrics=m.get("metrics", {}),
        thresholds={
            "temperature": p.predictor.temperature,
            "conformal_qhat": p.qhat,
            "tau_conf": p.tau_conf,
            "ood_scorer": p.ood_scorer,
            "tau_ood": p.tau_ood,
            "quality": p.meta["quality_thresholds"],
        },
    )


@router.post("/predict", response_model=PredictResponse, responses=_ERRORS)
def predict(request: Request, file: Annotated[UploadFile, File()]) -> PredictResponse:
    """Decision for one leaf photo: accepted / uncertain / rejected, with advice."""
    _, _, base = _run(request, file, with_cam=False)
    return PredictResponse(**base)


@router.post("/analyze", response_model=AnalyzeResponse, responses=_ERRORS)
def analyze(request: Request, file: Annotated[UploadFile, File()]) -> AnalyzeResponse:
    """The decision plus probabilities, OOD score, quality report and a CAM overlay."""
    pipeline, res, base = _run(request, file, with_cam=True)
    return AnalyzeResponse(
        **base,
        probabilities=res.probabilities,
        ood_score=res.ood_score,
        ood_threshold=pipeline.tau_ood,
        quality=res.quality,
        cam_png_base64=res.cam_png_base64,
        timings_ms=res.timings_ms,
    )
