"""Environment-backed runtime configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _positive_float(name: str, default: str) -> float:
    value = float(os.getenv(name, default))
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def _positive_int(name: str, default: str) -> int:
    value = int(os.getenv(name, default))
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


@dataclass(frozen=True)
class Settings:
    """Paths and limits shared by application adapters."""

    data_dir: Path
    database_path: Path
    chroma_path: Path
    forecast_base_url: str
    request_timeout_seconds: float
    max_file_bytes: int

    @classmethod
    def from_env(cls) -> Settings:
        data_dir = Path(os.getenv("TRAFFIC_KNOWLEDGE_DATA_DIR", "data")).expanduser()
        forecast_base_url = os.getenv(
            "TRAFFIC_FORECAST_BASE_URL", "http://127.0.0.1:18000"
        ).rstrip("/")
        if not forecast_base_url:
            raise ValueError("TRAFFIC_FORECAST_BASE_URL must not be empty")

        return cls(
            data_dir=data_dir,
            database_path=data_dir / "metadata.sqlite3",
            chroma_path=data_dir / "chroma",
            forecast_base_url=forecast_base_url,
            request_timeout_seconds=_positive_float(
                "TRAFFIC_REQUEST_TIMEOUT_SECONDS", "10"
            ),
            max_file_bytes=_positive_int("TRAFFIC_MAX_FILE_BYTES", "52428800"),
        )

    def ensure_directories(self) -> None:
        """Create only the project-owned runtime directories."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_path.mkdir(parents=True, exist_ok=True)
