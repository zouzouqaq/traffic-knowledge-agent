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


def _nonnegative_float(name: str, default: str) -> float:
    value = float(os.getenv(name, default))
    if value < 0:
        raise ValueError(f"{name} must be zero or greater")
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
    retrieval_vector_weight: float
    retrieval_bm25_weight: float
    embedding_model_name: str
    answer_mode: str
    deepseek_api_key: str | None
    deepseek_base_url: str
    deepseek_model: str
    deepseek_timeout_seconds: float
    deepseek_temperature: float
    deepseek_max_output_tokens: int

    @classmethod
    def from_env(cls) -> Settings:
        data_dir = Path(os.getenv("TRAFFIC_KNOWLEDGE_DATA_DIR", "data")).expanduser()
        forecast_base_url = os.getenv(
            "TRAFFIC_FORECAST_BASE_URL", "http://127.0.0.1:18000"
        ).rstrip("/")
        if not forecast_base_url:
            raise ValueError("TRAFFIC_FORECAST_BASE_URL must not be empty")
        vector_weight = _nonnegative_float("TRAFFIC_RETRIEVAL_VECTOR_WEIGHT", "0.6")
        bm25_weight = _nonnegative_float("TRAFFIC_RETRIEVAL_BM25_WEIGHT", "0.4")
        if vector_weight + bm25_weight <= 0:
            raise ValueError("at least one retrieval weight must be greater than zero")
        answer_mode = os.getenv("TRAFFIC_ANSWER_MODE", "evidence").strip().lower()
        if answer_mode not in {"evidence", "deepseek"}:
            raise ValueError("TRAFFIC_ANSWER_MODE must be evidence or deepseek")
        deepseek_api_key = os.getenv("DEEPSEEK_API_KEY", "").strip() or None
        if answer_mode == "deepseek" and deepseek_api_key is None:
            raise ValueError("DEEPSEEK_API_KEY is required in deepseek mode")
        deepseek_base_url = os.getenv(
            "DEEPSEEK_BASE_URL", "https://api.deepseek.com"
        ).strip().rstrip("/")
        if not deepseek_base_url:
            raise ValueError("DEEPSEEK_BASE_URL must not be empty")
        deepseek_model = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash").strip()
        if not deepseek_model:
            raise ValueError("DEEPSEEK_MODEL must not be empty")
        embedding_model_name = os.getenv(
            "TRAFFIC_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5"
        ).strip()
        if not embedding_model_name:
            raise ValueError("TRAFFIC_EMBEDDING_MODEL must not be empty")

        return cls(
            data_dir=data_dir,
            database_path=data_dir / "metadata.sqlite3",
            chroma_path=data_dir / "chroma",
            forecast_base_url=forecast_base_url,
            request_timeout_seconds=_positive_float(
                "TRAFFIC_REQUEST_TIMEOUT_SECONDS", "10"
            ),
            max_file_bytes=_positive_int("TRAFFIC_MAX_FILE_BYTES", "52428800"),
            retrieval_vector_weight=vector_weight,
            retrieval_bm25_weight=bm25_weight,
            embedding_model_name=embedding_model_name,
            answer_mode=answer_mode,
            deepseek_api_key=deepseek_api_key,
            deepseek_base_url=deepseek_base_url,
            deepseek_model=deepseek_model,
            deepseek_timeout_seconds=_positive_float("DEEPSEEK_TIMEOUT_SECONDS", "20"),
            deepseek_temperature=_nonnegative_float("DEEPSEEK_TEMPERATURE", "0.2"),
            deepseek_max_output_tokens=_positive_int("DEEPSEEK_MAX_OUTPUT_TOKENS", "800"),
        )

    def ensure_directories(self) -> None:
        """Create only the project-owned runtime directories."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_path.mkdir(parents=True, exist_ok=True)
