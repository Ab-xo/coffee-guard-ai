"""Response schemas (shown in the OpenAPI docs at /docs)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(examples=["ok"])
    model_loaded: bool
    model_name: str | None = None


class PredictResponse(BaseModel):
    label: str = Field(examples=["Leaf Rust"])
    confidence: float = Field(ge=0, le=1, description="Probability of the predicted class.")
    probabilities: dict[str, float]
    model_name: str
    latency_ms: float


class ErrorResponse(BaseModel):
    error: str = Field(examples=["unsupported_media_type"])
    detail: str
