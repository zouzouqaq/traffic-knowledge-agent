"""FastAPI surface for traffic documents, retrieval, Agent chat and metrics."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, File, Request, Response, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

from traffic_knowledge.api.dependencies import (
    ApiDependencies,
    get_api_dependencies,
    sanitize_filename,
)
from traffic_knowledge.domain.document import DocumentValidationError
from traffic_knowledge.ingestion.service import IngestionError
from traffic_knowledge.integrations.metrics_snapshot import MetricsSnapshotError

SUPPORTED_DOCUMENT_SUFFIXES = {".md", ".pdf", ".docx"}


class SearchRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)


class ChatRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    question: str = Field(min_length=1, max_length=4000)
    forecast_model: str | None = Field(default=None, min_length=1, max_length=64)
    forecast_inputs: list | None = None

    @field_validator("forecast_inputs")
    @classmethod
    def validate_forecast_inputs(cls, value):
        if value is not None and not value:
            raise ValueError("forecast_inputs must not be empty")
        return value


def _error_response(
    status_code: int,
    code: str,
    message: str,
    details: dict[str, Any] | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": code,
                "message": message,
                "details": details or {},
            }
        },
    )


def _require(service, name: str):
    if service is None:
        raise ServiceUnavailableError(name)
    return service


class ServiceUnavailableError(RuntimeError):
    def __init__(self, service: str) -> None:
        self.service = service
        super().__init__(f"{service} dependency is not configured")


def _encode(value):
    if is_dataclass(value):
        value = asdict(value)
    return jsonable_encoder(value)


app = FastAPI(title="Traffic Knowledge Agent API", version="0.1.0")


@app.exception_handler(RequestValidationError)
async def request_validation_handler(
    request: Request, error: RequestValidationError
) -> JSONResponse:
    del request
    return _error_response(
        422,
        "REQUEST_VALIDATION_FAILED",
        "request data is invalid",
        {"errors": jsonable_encoder(error.errors())},
    )


@app.exception_handler(DocumentValidationError)
async def document_validation_handler(
    request: Request, error: DocumentValidationError
) -> JSONResponse:
    del request
    status = {
        "DOCUMENT_TYPE_UNSUPPORTED": 415,
        "DOCUMENT_TOO_LARGE": 413,
        "DOCUMENT_NOT_FOUND": 404,
    }.get(error.code, 400)
    return _error_response(status, error.code, error.message)


@app.exception_handler(IngestionError)
async def ingestion_error_handler(request: Request, error: IngestionError) -> JSONResponse:
    del request
    return _error_response(503, error.code, error.message)


@app.exception_handler(ServiceUnavailableError)
async def unavailable_handler(
    request: Request, error: ServiceUnavailableError
) -> JSONResponse:
    del request
    return _error_response(
        503,
        "DEPENDENCY_UNAVAILABLE",
        f"{error.service} dependency is not configured",
    )


@app.exception_handler(StarletteHTTPException)
async def http_error_handler(
    request: Request, error: StarletteHTTPException
) -> JSONResponse:
    del request
    code = {
        404: "HTTP_NOT_FOUND",
        405: "HTTP_METHOD_NOT_ALLOWED",
    }.get(error.status_code, "HTTP_ERROR")
    return _error_response(error.status_code, code, str(error.detail))


@app.exception_handler(Exception)
async def internal_error_handler(request: Request, error: Exception) -> JSONResponse:
    del request, error
    return _error_response(
        500,
        "INTERNAL_SERVER_ERROR",
        "internal server error",
    )


@app.get("/health")
def health(
    dependencies: Annotated[ApiDependencies, Depends(get_api_dependencies)],
):
    configured = {
        "metadata": dependencies.document_service is not None,
        "retrieval": dependencies.retriever is not None,
        "forecast": dependencies.agent_graph is not None,
    }
    states = {
        name: bool(configured[name] and dependencies.health_states.get(name, False))
        for name in configured
    }
    return {
        "status": "ok" if states and all(states.values()) else "degraded",
        "dependencies": states,
    }


@app.post("/documents", status_code=201, responses={200: {"description": "Duplicate"}})
async def upload_document(
    response: Response,
    file: Annotated[UploadFile, File()],
    dependencies: Annotated[ApiDependencies, Depends(get_api_dependencies)],
):
    try:
        filename = sanitize_filename(file.filename or "")
    except ValueError as error:
        raise DocumentValidationError("DOCUMENT_TYPE_UNSUPPORTED", str(error)) from error
    if Path(filename).suffix.lower() not in SUPPORTED_DOCUMENT_SUFFIXES:
        raise DocumentValidationError("DOCUMENT_TYPE_UNSUPPORTED", filename)
    content = await file.read(dependencies.max_file_bytes + 1)
    if len(content) > dependencies.max_file_bytes:
        raise DocumentValidationError("DOCUMENT_TOO_LARGE", filename)
    service = _require(dependencies.document_service, "document_service")
    result = await run_in_threadpool(service.ingest_upload, filename, content)
    response.status_code = 200 if result.duplicate else 201
    return _encode(result)


@app.get("/documents")
def list_documents(
    dependencies: Annotated[ApiDependencies, Depends(get_api_dependencies)],
):
    service = _require(dependencies.document_service, "document_service")
    return {"documents": _encode(service.list_documents())}


@app.delete("/documents/{document_id}", status_code=204)
def delete_document(
    document_id: str,
    dependencies: Annotated[ApiDependencies, Depends(get_api_dependencies)],
):
    service = _require(dependencies.document_service, "document_service")
    if not service.delete_document(document_id):
        raise DocumentValidationError("DOCUMENT_NOT_FOUND", document_id)
    return Response(status_code=204)


@app.post("/retrieval/search")
def search(
    payload: SearchRequest,
    dependencies: Annotated[ApiDependencies, Depends(get_api_dependencies)],
):
    retriever = _require(dependencies.retriever, "retriever")
    return {"hits": _encode(retriever.search(payload.query, payload.top_k))}


@app.post("/chat")
def chat(
    payload: ChatRequest,
    dependencies: Annotated[ApiDependencies, Depends(get_api_dependencies)],
):
    graph = _require(dependencies.agent_graph, "agent_graph")
    request = payload.model_dump(exclude_none=True)
    try:
        state = graph.invoke(request)
    except ValueError as error:
        return _error_response(400, "AGENT_REQUEST_INVALID", str(error))
    return _encode(state["response"])


@app.get("/benchmarks/latest")
def latest_benchmark(
    dependencies: Annotated[ApiDependencies, Depends(get_api_dependencies)],
):
    repository = _require(dependencies.metrics_repository, "metrics_repository")
    try:
        snapshot = repository.load(dependencies.metrics_path)
    except FileNotFoundError as error:
        return _error_response(404, "BENCHMARK_NOT_FOUND", str(error))
    except MetricsSnapshotError as error:
        if isinstance(error.__cause__, FileNotFoundError):
            return _error_response(404, "BENCHMARK_NOT_FOUND", str(error.__cause__))
        return _error_response(500, error.code, error.message)
    return _encode(snapshot)
