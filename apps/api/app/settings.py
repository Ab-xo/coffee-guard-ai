"""API settings, read from environment variables (or a .env file)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    model_bundle: Path = Path("artifacts/models/coffeeguard-effv2b0-v1")
    max_upload_mb: float = 10.0
    max_pixels: int = 100_000_000  # the dataset's largest photo is 108 MP; refuse beyond
    ort_threads: int | None = None
    cors_origins: list[str] = ["http://localhost:8501"]
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
