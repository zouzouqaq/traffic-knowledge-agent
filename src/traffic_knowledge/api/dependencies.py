"""Dependency container for the FastAPI transport layer."""

from __future__ import annotations

import html
import os
import re
import tempfile
import threading
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import httpx

from traffic_knowledge.application.agent_graph import (
    AgentDependencies,
    build_agent_graph,
)
from traffic_knowledge.application.question_answering import QuestionAnsweringService
from traffic_knowledge.ingestion.repository import DocumentRepository
from traffic_knowledge.ingestion.service import IngestionService
from traffic_knowledge.integrations.forecast_client import ForecastClient
from traffic_knowledge.integrations.metrics_snapshot import MetricsSnapshotRepository
from traffic_knowledge.retrieval.bm25 import Bm25Index
from traffic_knowledge.retrieval.hybrid import HybridRetriever
from traffic_knowledge.settings import Settings


@dataclass(frozen=True)
class ApiDependencies:
    """Runtime services consumed by thin HTTP handlers."""

    document_service: object | None = None
    retriever: object | None = None
    agent_graph: object | None = None
    metrics_repository: object | None = None
    metrics_path: Path = Path("artifacts/metrics_snapshot.json")
    max_file_bytes: int = 50 * 1024 * 1024
    health_states: dict[str, bool] = field(
        default_factory=lambda: {
            "metadata": False,
            "retrieval": False,
            "forecast": False,
        }
    )

    def __post_init__(self) -> None:
        if self.max_file_bytes <= 0:
            raise ValueError("max_file_bytes must be greater than zero")


class DocumentManagementService:
    """Bridge uploaded bytes to the existing path-based ingestion services."""

    def __init__(
        self,
        ingestion_service: IngestionService,
        repository: DocumentRepository,
        vector_index,
        bm25_index: Bm25Index,
        staging_dir: Path,
    ) -> None:
        self.ingestion_service = ingestion_service
        self.repository = repository
        self.vector_index = vector_index
        self.bm25_index = bm25_index
        self.staging_dir = Path(staging_dir)
        self._lock = threading.RLock()

    def ingest_upload(self, filename: str, content: bytes):
        safe_name = sanitize_filename(filename)
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        with self._lock, tempfile.TemporaryDirectory(dir=self.staging_dir) as directory:
            path = Path(directory) / safe_name
            path.write_bytes(content)
            result = self.ingestion_service.ingest(path)
            self._rebuild_bm25()
            return result

    def list_documents(self):
        return self.repository.list_documents()

    def delete_document(self, document_id: str) -> bool:
        with self._lock:
            if self.repository.find_by_id(document_id) is None:
                return False
            self.vector_index.delete(document_id)
            deleted = self.repository.delete(document_id)
            self._rebuild_bm25()
            return deleted

    def _rebuild_bm25(self) -> None:
        self.bm25_index.rebuild(self.repository.list_all_chunks())


class EvidenceOnlyChatModel:
    """Offline MVP responder that quotes the highest-ranked evidence."""

    _EVIDENCE = re.compile(r"<evidence[^>]*>\s*(.*?)\s*</evidence>", re.DOTALL)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        del system_prompt
        match = self._EVIDENCE.search(user_prompt)
        if match is None:
            return "现有资料不足以回答该问题。"
        excerpt = " ".join(html.unescape(match.group(1)).split())[:240]
        first_statement = re.split(
            r"(?<=[.!?])\s+|(?<=[\u3002\uff01\uff1f])\s*",
            excerpt,
            maxsplit=1,
        )[0].rstrip(".!?\u3002\uff01\uff1f")
        return f"{first_statement} [S1]."


class RuleBasedIntentModel:
    """Deterministic MVP router; it cannot trigger arbitrary tools."""

    def classify(self, question: str) -> str:
        lowered = question.lower()
        has_forecast = any(word in lowered for word in ("预测", "未来", "下一"))
        has_metrics = any(
            word in lowered
            for word in ("指标", "mae", "rmse", "mape", "对比", "比较")
        )
        if "综合" in lowered or (has_forecast and has_metrics):
            return "combined"
        if has_forecast:
            return "forecast"
        if has_metrics:
            return "metrics"
        return "knowledge"


def sanitize_filename(filename: str) -> str:
    normalized = filename.replace("\\", "/")
    safe_name = normalized.rsplit("/", 1)[-1].strip()
    if not safe_name or safe_name in {".", ".."}:
        raise ValueError("filename must not be empty")
    return safe_name


def _forecast_is_healthy(base_url: str, timeout_seconds: float) -> bool:
    try:
        response = httpx.get(
            f"{base_url}/health",
            timeout=min(timeout_seconds, 1.0),
        )
        return response.status_code == 200
    except httpx.HTTPError:
        return False


def _build_default_dependencies() -> ApiDependencies:
    settings = Settings.from_env()
    settings.ensure_directories()
    repository = DocumentRepository(settings.database_path)
    repository.initialize()

    from sentence_transformers import SentenceTransformer

    from traffic_knowledge.retrieval.vector import ChromaVectorIndex

    embedding_model = SentenceTransformer(
        settings.embedding_model_name,
        device="cpu",
        local_files_only=True,
    )

    def filename_resolver(document_id: str) -> str:
        record = repository.find_by_id(document_id)
        return record.filename if record is not None else document_id

    vector_index = ChromaVectorIndex(
        embedding_model=embedding_model,
        chroma_path=settings.chroma_path,
        filename_resolver=filename_resolver,
    )
    bm25_index = Bm25Index(filename_resolver=filename_resolver)
    bm25_index.rebuild(repository.list_all_chunks())
    retriever = HybridRetriever(
        vector_retriever=vector_index,
        bm25_retriever=bm25_index,
        vector_weight=settings.retrieval_vector_weight,
        bm25_weight=settings.retrieval_bm25_weight,
    )
    ingestion_service = IngestionService(
        repository=repository,
        vector_index=vector_index,
        max_file_bytes=settings.max_file_bytes,
    )
    document_service = DocumentManagementService(
        ingestion_service=ingestion_service,
        repository=repository,
        vector_index=vector_index,
        bm25_index=bm25_index,
        staging_dir=settings.data_dir / "uploads",
    )
    qa_service = QuestionAnsweringService(
        retriever=retriever,
        chat_model=EvidenceOnlyChatModel(),
    )
    forecast_client = ForecastClient(
        base_url=settings.forecast_base_url,
        timeout_seconds=settings.request_timeout_seconds,
    )
    metrics_repository = MetricsSnapshotRepository()
    metrics_path = Path(
        os.getenv("TRAFFIC_METRICS_PATH", "artifacts/metrics_snapshot.json")
    ).expanduser()
    agent_graph = build_agent_graph(
        AgentDependencies(
            intent_model=RuleBasedIntentModel(),
            qa_service=qa_service,
            forecast_client=forecast_client,
            metrics_repository=metrics_repository,
            metrics_path=metrics_path,
        )
    )
    return ApiDependencies(
        document_service=document_service,
        retriever=retriever,
        agent_graph=agent_graph,
        metrics_repository=metrics_repository,
        metrics_path=metrics_path,
        max_file_bytes=settings.max_file_bytes,
        health_states={
            "metadata": True,
            "retrieval": True,
            "forecast": _forecast_is_healthy(
                settings.forecast_base_url,
                settings.request_timeout_seconds,
            ),
        },
    )


@lru_cache(maxsize=1)
def get_api_dependencies() -> ApiDependencies:
    """Build real project-owned services once, on the first API request."""
    return _build_default_dependencies()
