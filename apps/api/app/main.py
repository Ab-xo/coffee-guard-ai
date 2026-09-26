"""CoffeeGuard API.

Run from the repo root:
    uv run uvicorn app.main:app --app-dir apps/api
(the model bundle comes from MODEL_BUNDLE, default artifacts/models/coffeeguard-effv2b0-v1)
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.services.images import ApiError
from app.settings import Settings, get_settings

log = logging.getLogger("coffeeguard.api")


class JsonFormatter(logging.Formatter):
    """One JSON object per line: easy to grep and to ship to any log store."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "msg": record.getMessage(),
        }
        entry.update(getattr(record, "fields", {}))
        return json.dumps(entry)


def _setup_logging(level: str) -> None:
    if not any(isinstance(h.formatter, JsonFormatter) for h in log.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        log.addHandler(handler)
    log.setLevel(level.upper())
    log.propagate = False


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    _setup_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        from coffeeguard.inference.pipeline import Pipeline

        # Load once per process; a missing or incomplete bundle fails start-up loudly
        # rather than returning 503 on every request.
        app.state.pipeline = Pipeline(settings.model_bundle, threads=settings.ort_threads)
        log.info("model loaded", extra={"fields": {"bundle": str(settings.model_bundle)}})
        yield
        app.state.pipeline = None

    app = FastAPI(
        title="CoffeeGuard AI",
        description="Coffee leaf disease classification (Healthy, Cercospora, Leaf Rust, Phoma) "
        "with photo-quality and out-of-distribution checks.",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"],
    )

    @app.middleware("http")
    async def request_id_and_log(request: Request, call_next):
        rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:16]
        request.state.request_id = rid
        t0 = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        fields = {
            "request_id": rid,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "ms": round((time.perf_counter() - t0) * 1000, 1),
            **getattr(request.state, "log_extra", {}),
        }
        log.info("request", extra={"fields": fields})
        return response

    @app.exception_handler(ApiError)
    async def _api_error(request: Request, exc: ApiError) -> JSONResponse:
        request.state.log_extra = {"error": exc.error}
        return JSONResponse(
            status_code=exc.status_code, content={"error": exc.error, "detail": exc.detail}
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422, content={"error": "invalid_request", "detail": str(exc.errors())}
        )

    app.include_router(router)
    return app


app = create_app()
