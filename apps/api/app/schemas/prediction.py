"""Response schemas (shown in the OpenAPI docs at /docs)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(examples=["ok"])
    model_loaded: bool
    model_name: str | None = None
    model_version: str | None = None


class ModelInfoResponse(BaseModel):
    name: str
    version: str
    architecture: str
    classes: list[str]
    image_size: int
    data_fingerprint: str | None
    source_run: str | None
    metrics: dict[str, float | list[float]]
    thresholds: dict[str, float | str | dict[str, float]]


class PredictResponse(BaseModel):
    status: Literal["accepted", "uncertain", "rejected"] = Field(
        description="accepted: one confident class; uncertain: see prediction_set; rejected: "
        "see reason."
    )
    reason: Literal["low_quality", "ood", "low_confidence"] | None = None
    label: str | None = Field(
        None, examples=["Leaf Rust"], description="Top class (not when rejected)."
    )
    confidence: float | None = Field(
        None, ge=0, le=1, description="Calibrated probability of label."
    )
    prediction_set: list[str] = Field(
        default_factory=list, description="Classes the true one is among (98% conformal set)."
    )
    issues: list[str] = Field(default_factory=list, examples=[["too_dark"]])
    advice: list[str] = Field(default_factory=list)
    model_version: str
    latency_ms: float


class AnalyzeResponse(PredictResponse):
    probabilities: dict[str, float] | None = None
    ood_score: float | None = Field(None, description="Higher = less like the training leaves.")
    ood_threshold: float
    quality: dict[str, float]
    cam_png_base64: str | None = Field(
        None, description="PNG (base64) of the model input with the class activation map overlaid."
    )
    timings_ms: dict[str, float]


class ErrorResponse(BaseModel):
    error: str = Field(examples=["unsupported_media_type"])
    detail: str
