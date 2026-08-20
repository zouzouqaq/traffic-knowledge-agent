import threading
from dataclasses import dataclass
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from traffic_knowledge.api.app import app
from traffic_knowledge.api.dependencies import (
    ApiDependencies,
    DocumentManagementService,
    EvidenceOnlyChatModel,
    get_api_dependencies,
)
from traffic_knowledge.domain.agent import AgentError, AgentResponse, ToolCallRecord
from traffic_knowledge.domain.document import DocumentValidationError
from traffic_knowledge.domain.retrieval import SearchHit
from traffic_knowledge.ingestion.repository import DocumentRecord
from traffic_knowledge.ingestion.service import IngestionResult
from traffic_knowledge.integrations.metrics_snapshot import (
    ForecastHorizon,
    MetricsSnapshot,
    MetricsSnapshotRepository,
    ModelMetrics,
)
from traffic_knowledge.retrieval.citations import Citation


class FakeDocumentService:
    def __init__(self):
        self.uploads = []
        self.deleted = []
        self.duplicate = False
        self.upload_error = None
        self.documents = [
            DocumentRecord(
                document_id="doc-1",
                sha256="abc",
                filename="guide.md",
                status="indexed",
                error_code=None,
                created_at="2026-08-20T01:00:00+00:00",
                updated_at="2026-08-20T01:00:00+00:00",
            )
        ]
        self.ingest_threads = []

    def ingest_upload(self, filename, content):
        self.ingest_threads.append(threading.current_thread().name)
        self.uploads.append((filename, content))
        if self.upload_error is not None:
            raise self.upload_error
        return IngestionResult(
            document_id="doc-1",
            sha256="abc",
            status="indexed",
            chunk_count=2,
            elapsed_ms=4.2,
            duplicate=self.duplicate,
        )

    def list_documents(self):
        return tuple(self.documents)

    def delete_document(self, document_id):
        self.deleted.append(document_id)
        return document_id == "doc-1"


class FakeRetriever:
    def search(self, query, top_k):
        return (
            SearchHit(
                chunk_id="chunk-1",
                document_id="doc-1",
                text="GRU uses historical traffic flow.",
                location="paragraph 1",
                filename="guide.md",
                channels=("vector", "bm25"),
                ranks=(("vector", 1), ("bm25", 1)),
                score=0.016,
            ),
        )[:top_k]


class FakeAgentGraph:
    def __init__(self):
        self.requests = []
        self.response = AgentResponse(
            answer="GRU 使用历史交通流 [S1]。",
            citations=(
                Citation(
                    label="S1",
                    document_id="doc-1",
                    filename="guide.md",
                    location="paragraph 1",
                    chunk_id="chunk-1",
                    excerpt="GRU 使用历史交通流。",
                ),
            ),
            tool_calls=(
                ToolCallRecord(
                    name="search_traffic_knowledge",
                    arguments={"question": "GRU 是什么"},
                    duration_ms=1.2,
                    success=True,
                ),
            ),
            partial=False,
            errors=(),
        )

    def invoke(self, request):
        self.requests.append(request)
        return {"response": self.response}


class FakeMetricsRepository:
    def __init__(self):
        self.missing = False

    def load(self, path):
        if self.missing:
            raise FileNotFoundError(path)
        return MetricsSnapshot(
            schema_version="1.0",
            dataset="PEMS04",
            split="test",
            horizon=ForecastHorizon(steps=12, interval_minutes=5),
            created_at="2026-08-19T13:30:00+08:00",
            environment={"device": "RTX 4090"},
            models=(ModelMetrics(name="gru", mae=27.1, rmse=41.0, mape=23.0),),
        )


@dataclass
class ApiFixture:
    client: TestClient
    documents: FakeDocumentService
    agent: FakeAgentGraph
    metrics: FakeMetricsRepository


@pytest.fixture
def api():
    documents = FakeDocumentService()
    agent = FakeAgentGraph()
    metrics = FakeMetricsRepository()
    dependencies = ApiDependencies(
        document_service=documents,
        retriever=FakeRetriever(),
        agent_graph=agent,
        metrics_repository=metrics,
        metrics_path=Path("metrics.json"),
        max_file_bytes=32,
        health_states={"metadata": True, "retrieval": True, "forecast": True},
    )
    app.dependency_overrides[get_api_dependencies] = lambda: dependencies
    try:
        with TestClient(app) as client:
            yield ApiFixture(client, documents, agent, metrics)
    finally:
        app.dependency_overrides.clear()


def test_health_reports_dependency_states(api):
    response = api.client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "dependencies": {"metadata": True, "retrieval": True, "forecast": True},
    }


def test_health_reports_degraded_when_one_dependency_is_unavailable(api):
    dependencies = get_api_dependencies_override(api, forecast=False)
    app.dependency_overrides[get_api_dependencies] = lambda: dependencies

    response = api.client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "degraded"
    assert response.json()["dependencies"]["forecast"] is False


def test_health_cannot_report_ok_when_services_are_not_configured(api):
    dependencies = ApiDependencies(
        health_states={"metadata": True, "retrieval": True, "forecast": True}
    )
    app.dependency_overrides[get_api_dependencies] = lambda: dependencies

    response = api.client.get("/health")

    assert response.json()["status"] == "degraded"
    assert response.json()["dependencies"] == {
        "metadata": False,
        "retrieval": False,
        "forecast": False,
    }


def get_api_dependencies_override(api, **health_states):
    states = {"metadata": True, "retrieval": True, "forecast": True}
    states.update(health_states)
    return ApiDependencies(
        document_service=api.documents,
        retriever=FakeRetriever(),
        agent_graph=api.agent,
        metrics_repository=api.metrics,
        metrics_path=Path("metrics.json"),
        max_file_bytes=32,
        health_states=states,
    )


def test_uploads_supported_document_and_reports_duplicate(api):
    created = api.client.post(
        "/documents",
        files={"file": ("guide.md", b"# traffic", "text/markdown")},
    )
    api.documents.duplicate = True
    duplicate = api.client.post(
        "/documents",
        files={"file": ("guide.md", b"# traffic", "text/markdown")},
    )

    assert created.status_code == 201
    assert created.json()["duplicate"] is False
    assert duplicate.status_code == 200
    assert duplicate.json()["duplicate"] is True
    assert api.documents.uploads[0] == ("guide.md", b"# traffic")
    assert "worker" in api.documents.ingest_threads[0].lower()


def test_upload_sanitizes_windows_style_path(api):
    response = api.client.post(
        "/documents",
        files={"file": ("..\\guide.md", b"# traffic", "text/markdown")},
    )

    assert response.status_code == 201
    assert api.documents.uploads[0][0] == "guide.md"


def test_upload_rejects_unsupported_type_with_consistent_error(api):
    response = api.client.post(
        "/documents",
        files={"file": ("payload.exe", b"binary", "application/octet-stream")},
    )

    assert response.status_code == 415
    assert response.json() == {
        "error": {
            "code": "DOCUMENT_TYPE_UNSUPPORTED",
            "message": "payload.exe",
            "details": {},
        }
    }


def test_upload_rejects_oversized_content_before_delegating(api):
    response = api.client.post(
        "/documents",
        files={"file": ("large.md", b"x" * 33, "text/markdown")},
    )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "DOCUMENT_TOO_LARGE"
    assert api.documents.uploads == []


def test_lists_and_deletes_documents(api):
    listed = api.client.get("/documents")
    deleted = api.client.delete("/documents/doc-1")
    missing = api.client.delete("/documents/missing")

    assert listed.status_code == 200
    assert listed.json()["documents"][0]["filename"] == "guide.md"
    assert deleted.status_code == 204
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "DOCUMENT_NOT_FOUND"


def test_search_validates_request_and_returns_ranked_hits(api):
    invalid = api.client.post("/retrieval/search", json={"query": "", "top_k": 5})
    valid = api.client.post(
        "/retrieval/search",
        json={"query": "GRU traffic", "top_k": 3},
    )

    assert invalid.status_code == 422
    assert invalid.json()["error"]["code"] == "REQUEST_VALIDATION_FAILED"
    assert valid.status_code == 200
    assert valid.json()["hits"][0]["chunk_id"] == "chunk-1"


def test_framework_404_uses_consistent_error_envelope(api):
    response = api.client.get("/does-not-exist")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "HTTP_NOT_FOUND"


def test_unexpected_backend_failure_uses_consistent_error_envelope(api):
    class BrokenRetriever:
        def search(self, query, top_k):
            raise RuntimeError("database password must not leak")

    dependencies = get_api_dependencies_override(api)
    dependencies = ApiDependencies(
        document_service=dependencies.document_service,
        retriever=BrokenRetriever(),
        agent_graph=dependencies.agent_graph,
        metrics_repository=dependencies.metrics_repository,
        metrics_path=dependencies.metrics_path,
        max_file_bytes=dependencies.max_file_bytes,
        health_states=dependencies.health_states,
    )
    app.dependency_overrides[get_api_dependencies] = lambda: dependencies

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/retrieval/search",
            json={"query": "traffic", "top_k": 3},
        )

    assert response.status_code == 500
    assert response.json() == {
        "error": {
            "code": "INTERNAL_SERVER_ERROR",
            "message": "internal server error",
            "details": {},
        }
    }


def test_chat_returns_citations_and_tool_trace(api):
    response = api.client.post("/chat", json={"question": "GRU 是什么"})

    assert response.status_code == 200
    assert response.json()["answer"].endswith("[S1]。")
    assert response.json()["citations"][0]["filename"] == "guide.md"
    assert response.json()["tool_calls"][0]["name"] == "search_traffic_knowledge"
    assert api.agent.requests[0]["question"] == "GRU 是什么"


def test_chat_rejects_whitespace_only_forecast_model(api):
    response = api.client.post(
        "/chat",
        json={"question": "预测", "forecast_model": "   ", "forecast_inputs": [[[[1]]]]},
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "REQUEST_VALIDATION_FAILED"


def test_chat_preserves_partial_agent_response(api):
    api.agent.response = AgentResponse(
        answer="知识检索成功, 但预测服务不可用。",
        citations=(),
        tool_calls=(
            ToolCallRecord(
                name="run_traffic_forecast",
                arguments={"model": "gru"},
                duration_ms=3.0,
                success=False,
                error_code="FORECAST_UNAVAILABLE",
            ),
        ),
        partial=True,
        errors=(
            AgentError(
                code="FORECAST_UNAVAILABLE",
                message="service down",
                tool="run_traffic_forecast",
            ),
        ),
    )

    response = api.client.post("/chat", json={"question": "综合分析"})

    assert response.status_code == 200
    assert response.json()["partial"] is True
    assert response.json()["errors"][0]["code"] == "FORECAST_UNAVAILABLE"


def test_latest_benchmark_returns_context_and_handles_missing_file(api):
    found = api.client.get("/benchmarks/latest")
    api.metrics.missing = True
    missing = api.client.get("/benchmarks/latest")

    assert found.status_code == 200
    assert found.json()["dataset"] == "PEMS04"
    assert found.json()["models"][0]["mae"] == pytest.approx(27.1)
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "BENCHMARK_NOT_FOUND"


def test_latest_benchmark_maps_real_missing_snapshot_to_not_found(api, tmp_path):
    dependencies = get_api_dependencies_override(api)
    dependencies = ApiDependencies(
        document_service=dependencies.document_service,
        retriever=dependencies.retriever,
        agent_graph=dependencies.agent_graph,
        metrics_repository=MetricsSnapshotRepository(),
        metrics_path=tmp_path / "missing.json",
        max_file_bytes=dependencies.max_file_bytes,
        health_states=dependencies.health_states,
    )
    app.dependency_overrides[get_api_dependencies] = lambda: dependencies

    response = api.client.get("/benchmarks/latest")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "BENCHMARK_NOT_FOUND"


def test_chat_maps_agent_argument_errors_to_consistent_error(api):
    class InvalidAgent:
        def invoke(self, request):
            raise ValueError("forecast_inputs are required")

    dependencies = get_api_dependencies_override(api)
    dependencies = ApiDependencies(
        document_service=dependencies.document_service,
        retriever=dependencies.retriever,
        agent_graph=InvalidAgent(),
        metrics_repository=dependencies.metrics_repository,
        metrics_path=dependencies.metrics_path,
        max_file_bytes=dependencies.max_file_bytes,
        health_states=dependencies.health_states,
    )
    app.dependency_overrides[get_api_dependencies] = lambda: dependencies

    response = api.client.post("/chat", json={"question": "预测下一小时"})

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "AGENT_REQUEST_INVALID"


def test_domain_errors_use_consistent_error_envelope(api):
    api.documents.upload_error = DocumentValidationError(
        "DOCUMENT_EMPTY", "guide.md"
    )

    response = api.client.post(
        "/documents",
        files={"file": ("guide.md", b"content", "text/markdown")},
    )

    assert response.status_code == 400
    assert response.json()["error"] == {
        "code": "DOCUMENT_EMPTY",
        "message": "guide.md",
        "details": {},
    }


def test_openapi_declares_created_upload_response():
    responses = app.openapi()["paths"]["/documents"]["post"]["responses"]

    assert "201" in responses


def test_offline_chat_model_places_citation_inside_sentence_boundary():
    model = EvidenceOnlyChatModel()

    answer = model.generate(
        "system",
        "<evidence label=\"S1\">MAE measures prediction error.</evidence>",
    )

    assert answer == "MAE measures prediction error [S1]."


def test_real_document_facade_ingests_lists_and_deletes(tmp_path):
    from traffic_knowledge.ingestion.repository import DocumentRepository
    from traffic_knowledge.ingestion.service import IngestionService

    class FakeVectorIndex:
        def __init__(self):
            self.chunks = []
            self.deleted = []

        def upsert(self, chunks):
            self.chunks.extend(chunks)

        def delete(self, document_id):
            self.deleted.append(document_id)

    class FakeBm25Index:
        def __init__(self):
            self.rebuilds = []

        def rebuild(self, chunks):
            self.rebuilds.append(tuple(chunks))

    repository = DocumentRepository(tmp_path / "metadata.sqlite3")
    repository.initialize()
    vector = FakeVectorIndex()
    bm25 = FakeBm25Index()
    ingestion = IngestionService(
        repository=repository,
        vector_index=vector,
        max_file_bytes=1024,
    )
    service = DocumentManagementService(
        ingestion_service=ingestion,
        repository=repository,
        vector_index=vector,
        bm25_index=bm25,
        staging_dir=tmp_path / "uploads",
    )

    result = service.ingest_upload("../guide.md", b"# Traffic\nGRU predicts flow.")
    documents = service.list_documents()
    deleted = service.delete_document(result.document_id)

    assert documents[0].filename == "guide.md"
    assert vector.chunks
    assert len(bm25.rebuilds) == 2
    assert vector.deleted == [result.document_id]
    assert deleted is True
    assert service.list_documents() == ()
