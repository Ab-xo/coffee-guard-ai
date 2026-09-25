"""CoffeeGuard API.

Run from the repo root:
    MODEL_BUNDLE=artifacts/models/<bundle> uv run uvicorn app.main:app --app-dir apps/api
"""

from __future__ import annotations

import logging
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


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        from coffeeguard.inference.predictor import Predictor

        # Load once per process; a missing bundle fails start-up loudly rather than
        # returning 503 on every request.
        app.state.predictor = Predictor(settings.model_bundle, threads=settings.ort_threads)
        log.info("Loaded model bundle %s", settings.model_bundle)
        yield
        app.state.predictor = None

    app = FastAPI(
        title="CoffeeGuard AI",
        description="Coffee leaf disease classification (Healthy, Cercospora, Leaf Rust, Phoma).",
        version="0.2.0",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.exception_handler(ApiError)
    async def _api_error(_: Request, exc: ApiError) -> JSONResponse:
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
