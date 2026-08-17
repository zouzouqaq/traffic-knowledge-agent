# Traffic Knowledge Agent MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a separate traffic-domain RAG and analysis Agent that ingests trusted documents, answers with citations, calls the existing forecasting service, and produces reproducible quality and performance reports.

**Architecture:** A Python 3.11 FastAPI application owns ingestion, SQLite metadata, Chroma vectors, BM25 retrieval and a LangGraph workflow. A Streamlit client calls the API. The existing `traffic-ops-agent` remains an external HTTP service and metrics artifact provider.

**Tech Stack:** Python 3.11, FastAPI, PyMuPDF, python-docx, sentence-transformers/FlagEmbedding, ChromaDB, rank-bm25, LangGraph, httpx, Streamlit, pytest, Ruff, psutil, optional Ragas.

## Global Constraints

- Work only under `/8t/usr/zhouh2024/projects/traffic-knowledge-agent`.
- Run interactive work in dedicated windows of the existing `tmux b` session.
- Do not access or modify other users' files or `/8t/usr/zhouh2024/zouz/STDN-main`.
- Do not use `systemctl`, restart Docker, or clean global Docker resources.
- The MVP must run without Docker, MySQL, Redis, Elasticsearch or MinIO.
- Default to CPU. Check `nvidia-smi` immediately before optional GPU use and use at most one idle GPU.
- Bind development services to `127.0.0.1`; access them with SSH port forwarding.
- Tests must not require a paid LLM API, a live forecasting service or a GPU.
- Raw documents, local indexes, secrets and benchmark run artifacts remain outside Git.
- Each task is one coherent review batch: write its tests first, implement the batch, then run focused tests, full tests and Ruff once.

---

## File Map

```text
traffic-knowledge-agent/
|-- .env.example                         # Safe configuration template
|-- .gitignore                           # Secrets, runtime data, indexes and artifacts
|-- README.md                            # Setup, commands, architecture and evidence
|-- pyproject.toml                       # Package and dependency groups
|-- streamlit_app.py                     # Thin demonstration client
|-- evaluation/
|   `-- questions.jsonl                  # At least 50 reviewed evaluation records
|-- scripts/
|   |-- ingest_documents.py              # Offline ingestion CLI
|   |-- evaluate_retrieval.py            # Vector/BM25/hybrid comparison CLI
|   |-- evaluate_rag.py                  # Citation and generated-answer evaluation CLI
|   `-- run_benchmark.py                 # Latency, throughput and memory CLI
|-- src/traffic_knowledge/
|   |-- __init__.py
|   |-- settings.py                      # Environment-backed paths and limits
|   |-- domain/
|   |   |-- document.py                  # Document/chunk models and validation errors
|   |   |-- retrieval.py                 # Search result and citation models
|   |   `-- agent.py                     # Agent state and response models
|   |-- ingestion/
|   |   |-- loaders.py                   # PDF/DOCX/Markdown parsing
|   |   |-- chunking.py                  # Section-aware bounded chunking
|   |   |-- repository.py                # SQLite metadata transactions
|   |   `-- service.py                   # Idempotent ingestion workflow
|   |-- retrieval/
|   |   |-- vector.py                    # Chroma adapter and embedding protocol
|   |   |-- bm25.py                      # Lexical index adapter
|   |   |-- hybrid.py                    # Reciprocal-rank fusion
|   |   `-- citations.py                 # Source numbering and excerpts
|   |-- integrations/
|   |   |-- forecast_client.py           # Typed forecast HTTP adapter
|   |   `-- metrics_snapshot.py          # Versioned benchmark JSON reader
|   |-- application/
|   |   |-- question_answering.py        # Grounded generation pipeline
|   |   `-- agent_graph.py               # LangGraph routing and tool limits
|   |-- api/
|   |   |-- dependencies.py              # Application assembly
|   |   `-- app.py                       # HTTP contracts and error mapping
|   `-- evaluation/
|       |-- dataset.py                    # Evaluation JSONL validation
|       |-- retrieval_metrics.py          # Hit@k, Recall@k and MRR
|       |-- answer_metrics.py             # Citation/tool/optional Ragas metrics
|       `-- performance.py                # Latency, throughput and memory capture
`-- tests/
    |-- fixtures/
    `-- test_*.py
```

## Task 1: Project Foundation and Configuration

**Files:**
- Create: `.gitignore`
- Create: `.env.example`
- Create: `pyproject.toml`
- Create: `README.md`
- Create: `src/traffic_knowledge/__init__.py`
- Create: `src/traffic_knowledge/settings.py`
- Create: `tests/test_environment.py`
- Create: `tests/test_settings.py`

**Interfaces:**
- Produces: `Settings.from_env() -> Settings`
- Produces: validated `data_dir`, `database_path`, `chroma_path`, `forecast_base_url`, `request_timeout_seconds`, `max_file_bytes`.

- [ ] **Step 1: Create foundation tests**

```python
from pathlib import Path

import traffic_knowledge
from traffic_knowledge.settings import Settings


def test_package_is_importable():
    assert traffic_knowledge.__version__


def test_settings_create_runtime_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAFFIC_KNOWLEDGE_DATA_DIR", str(tmp_path))
    settings = Settings.from_env()
    settings.ensure_directories()
    assert settings.database_path == tmp_path / "metadata.sqlite3"
    assert settings.chroma_path == tmp_path / "chroma"
    assert settings.chroma_path.is_dir()
```

- [ ] **Step 2: Run the tests and verify collection fails because the package does not exist**

Run: `pytest tests/test_environment.py tests/test_settings.py -v`
Expected: collection error containing `No module named 'traffic_knowledge'`.

- [ ] **Step 3: Add package metadata, dependencies, ignores and settings**

Use these dependency groups in `pyproject.toml`:

```toml
dependencies = [
  "fastapi>=0.115,<1",
  "httpx>=0.27,<1",
  "langgraph>=0.2,<1",
  "numpy>=1.26,<3",
  "pydantic>=2.9,<3",
  "pydantic-settings>=2.5,<3",
  "pymupdf>=1.24,<2",
  "python-docx>=1.1,<2",
  "python-multipart>=0.0.12,<1",
  "rank-bm25>=0.2.2,<1",
  "streamlit>=1.38,<2",
  "uvicorn>=0.30,<1",
]

[project.optional-dependencies]
retrieval = [
  "chromadb>=0.5,<1",
  "sentence-transformers>=3,<6",
]
evaluation = ["psutil>=6,<8", "ragas>=0.2,<1"]
dev = ["pytest>=8,<9", "pytest-cov>=5,<7", "ruff>=0.6,<1"]
```

Implement immutable settings:

```python
@dataclass(frozen=True)
class Settings:
    data_dir: Path
    database_path: Path
    chroma_path: Path
    forecast_base_url: str
    request_timeout_seconds: float
    max_file_bytes: int

    @classmethod
    def from_env(cls) -> "Settings": ...

    def ensure_directories(self) -> None: ...
```

`.gitignore` must exclude `.env`, `.venv/`, `data/`, `artifacts/`, `.pytest_cache/`, `.ruff_cache/`, `__pycache__/` and model files. `.env.example` must contain no real key.

- [ ] **Step 4: Install only foundation and dev dependencies, then verify**

Run: `python -m pip install -e ".[dev]"`
Run: `pytest tests/test_environment.py tests/test_settings.py -v`
Run: `ruff check .`
Expected: all tests and lint pass.

- [ ] **Step 5: Commit the foundation**

```bash
git add .gitignore .env.example pyproject.toml README.md src tests
git commit -m "chore: initialize traffic knowledge agent"
```

## Task 2: Domain Contracts and Document Loaders

**Files:**
- Create: `src/traffic_knowledge/domain/document.py`
- Create: `src/traffic_knowledge/ingestion/loaders.py`
- Create: `tests/fixtures/sample.md`
- Create: `tests/fixtures/sample.docx` through a pytest fixture, not as an opaque binary commit
- Create: `tests/fixtures/sample.pdf` through a pytest fixture, not as an opaque binary commit
- Create: `tests/test_document_loaders.py`

**Interfaces:**
- Produces: `SourceBlock(text: str, location: str, ordinal: int)`
- Produces: `ParsedDocument(filename: str, media_type: str, blocks: tuple[SourceBlock, ...])`
- Produces: `load_document(path: Path) -> ParsedDocument`
- Raises: `DocumentValidationError(code: str, message: str)`.

- [ ] **Step 1: Write loader contract tests**

Test Markdown headings map to locations such as `Introduction > Metrics`, DOCX headings produce the same form, and PDF blocks retain `page:1`. Also test unsupported extension, empty parsed content and encrypted/unreadable PDF error codes.

```python
def test_loads_markdown_with_heading_location(tmp_path):
    path = tmp_path / "guide.md"
    path.write_text("# Traffic\n## GRU\nShort-term model.", encoding="utf-8")
    parsed = load_document(path)
    assert parsed.blocks[0].location == "Traffic > GRU"
    assert parsed.blocks[0].text == "Short-term model."
```

- [ ] **Step 2: Run focused tests and verify failure**

Run: `pytest tests/test_document_loaders.py -v`
Expected: import error for `traffic_knowledge.domain.document`.

- [ ] **Step 3: Implement domain models and three loaders**

Use a suffix dispatch table, not nested string conditions:

```python
LOADERS = {
    ".md": _load_markdown,
    ".docx": _load_docx,
    ".pdf": _load_pdf,
}


def load_document(path: Path) -> ParsedDocument:
    loader = LOADERS.get(path.suffix.lower())
    if loader is None:
        raise DocumentValidationError("DOCUMENT_TYPE_UNSUPPORTED", path.suffix)
    parsed = loader(path)
    if not any(block.text.strip() for block in parsed.blocks):
        raise DocumentValidationError("DOCUMENT_EMPTY", path.name)
    return parsed
```

Normalize repeated whitespace but retain sentence content. Never execute macros, embedded objects or document instructions.

- [ ] **Step 4: Run batch verification and commit**

Run: `pytest tests/test_document_loaders.py -v`
Run: `pytest -v && ruff check .`
Expected: all pass.

```bash
git add src/traffic_knowledge/domain src/traffic_knowledge/ingestion tests
git commit -m "feat: parse traffic knowledge documents"
```

## Task 3: Stable Chunking and SQLite Metadata

**Files:**
- Create: `src/traffic_knowledge/ingestion/chunking.py`
- Create: `src/traffic_knowledge/ingestion/repository.py`
- Create: `tests/test_chunking.py`
- Create: `tests/test_document_repository.py`

**Interfaces:**
- Consumes: `ParsedDocument`, `SourceBlock` from Task 2.
- Produces: `DocumentChunk(chunk_id, document_id, text, location, ordinal, token_estimate)`.
- Produces: `chunk_document(document_id, parsed, max_characters, overlap_characters) -> tuple[DocumentChunk, ...]`.
- Produces: `DocumentRepository.initialize()`, `find_by_sha256()`, `begin_ingestion()`, `replace_chunks()`, `mark_indexed()`, `mark_failed()`, `delete()`.

- [ ] **Step 1: Write tests for boundaries, stable IDs and transactions**

Verify chunks never exceed the configured bound except one indivisible sentence, overlap is bounded, identical input produces identical IDs, successful replacement is atomic, and failed ingestion can be retried.

- [ ] **Step 2: Verify tests fail before implementation**

Run: `pytest tests/test_chunking.py tests/test_document_repository.py -v`
Expected: missing module errors.

- [ ] **Step 3: Implement sentence-aware chunking and SQLite schema**

Use stable IDs:

```python
payload = f"{document_id}:{block.ordinal}:{chunk_ordinal}:{text}".encode("utf-8")
chunk_id = hashlib.sha256(payload).hexdigest()
```

Create tables with uniqueness and foreign keys:

```sql
CREATE TABLE documents (
  document_id TEXT PRIMARY KEY,
  sha256 TEXT NOT NULL UNIQUE,
  filename TEXT NOT NULL,
  status TEXT NOT NULL CHECK(status IN ('indexing','indexed','failed')),
  error_code TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE TABLE chunks (
  chunk_id TEXT PRIMARY KEY,
  document_id TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
  ordinal INTEGER NOT NULL,
  location TEXT NOT NULL,
  text TEXT NOT NULL,
  UNIQUE(document_id, ordinal)
);
```

Enable `PRAGMA foreign_keys=ON` for every connection and use explicit transactions.

- [ ] **Step 4: Verify and commit**

Run: `pytest tests/test_chunking.py tests/test_document_repository.py -v`
Run: `pytest -v && ruff check .`
Expected: all pass.

```bash
git add src/traffic_knowledge/ingestion tests
git commit -m "feat: add stable chunks and metadata repository"
```

## Task 4: Idempotent Ingestion Service

**Files:**
- Create: `src/traffic_knowledge/ingestion/service.py`
- Create: `scripts/ingest_documents.py`
- Create: `tests/test_ingestion_service.py`
- Create: `tests/test_ingest_documents_cli.py`

**Interfaces:**
- Consumes: loader, chunker and repository from Tasks 2-3.
- Consumes: `VectorIndex.upsert(chunks)` protocol, implemented by a fake in this task.
- Produces: `IngestionResult(document_id, sha256, status, chunk_count, elapsed_ms, duplicate)`.
- Produces: `IngestionService.ingest(path: Path) -> IngestionResult`.

- [ ] **Step 1: Write one end-to-end ingestion test and failure tests**

Cover identical-byte duplicate, same filename with different bytes, file exceeding `max_file_bytes`, index failure marked as failed, and retry after failure.

- [ ] **Step 2: Run focused tests and observe missing service failure**

Run: `pytest tests/test_ingestion_service.py tests/test_ingest_documents_cli.py -v`.

- [ ] **Step 3: Implement streaming SHA-256 and orchestration**

```python
def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()
```

The service sequence is hash -> duplicate lookup -> begin -> parse -> chunk -> metadata write -> vector upsert -> mark indexed. Catch known validation errors separately from unexpected indexing errors.

- [ ] **Step 4: Verify JSON CLI output and commit**

Run: `python scripts/ingest_documents.py --help`
Run: `pytest tests/test_ingestion_service.py tests/test_ingest_documents_cli.py -v`
Run: `pytest -v && ruff check .`.

```bash
git add src/traffic_knowledge/ingestion scripts tests
git commit -m "feat: add idempotent document ingestion"
```

## Task 5: Vector, BM25 and Hybrid Retrieval

**Files:**
- Create: `src/traffic_knowledge/domain/retrieval.py`
- Create: `src/traffic_knowledge/retrieval/vector.py`
- Create: `src/traffic_knowledge/retrieval/bm25.py`
- Create: `src/traffic_knowledge/retrieval/hybrid.py`
- Create: `tests/test_vector_index.py`
- Create: `tests/test_bm25_index.py`
- Create: `tests/test_hybrid_retrieval.py`

**Interfaces:**
- Produces: `SearchHit(chunk_id, text, location, filename, channels, ranks, score)`.
- Produces: `EmbeddingModel.encode(texts: Sequence[str]) -> np.ndarray` protocol.
- Produces: `ChromaVectorIndex.upsert/delete/search`.
- Produces: `Bm25Index.rebuild/search`.
- Produces: `HybridRetriever.search(query: str, top_k: int) -> tuple[SearchHit, ...]`.

- [ ] **Step 1: Write deterministic adapter and fusion tests**

Use fake embeddings so tests do not download a model. Assert deletion removes vector hits, BM25 favors exact technical terms, duplicate chunk IDs are fused once, ordering is deterministic, and invalid `top_k` is rejected.

- [ ] **Step 2: Run tests and verify missing module failures**

Run: `pytest tests/test_vector_index.py tests/test_bm25_index.py tests/test_hybrid_retrieval.py -v`.

- [ ] **Step 3: Implement adapters and weighted reciprocal-rank fusion**

```python
def reciprocal_rank_fusion(result_sets, weights, constant=60):
    scores = defaultdict(float)
    for channel, hits in result_sets.items():
        for rank, hit in enumerate(hits, start=1):
            scores[hit.chunk_id] += weights[channel] / (constant + rank)
    return scores
```

Default weights: vector `0.6`, BM25 `0.4`. Store them in settings and include channel ranks in every hit.

- [ ] **Step 4: Install retrieval dependencies and run one CPU smoke check**

Run: `python -m pip install -e ".[retrieval,dev]"`
Run a script encoding two short Chinese sentences with `BAAI/bge-small-zh-v1.5`; record download location and peak process memory, but do not add model files to Git.

- [ ] **Step 5: Verify and commit**

Run: `pytest tests/test_vector_index.py tests/test_bm25_index.py tests/test_hybrid_retrieval.py -v`
Run: `pytest -v && ruff check .`.

```bash
git add pyproject.toml src/traffic_knowledge/domain src/traffic_knowledge/retrieval tests
git commit -m "feat: add hybrid traffic knowledge retrieval"
```

## Task 6: Citations and Grounded Question Answering

**Files:**
- Create: `src/traffic_knowledge/retrieval/citations.py`
- Create: `src/traffic_knowledge/application/question_answering.py`
- Create: `tests/test_citations.py`
- Create: `tests/test_question_answering.py`

**Interfaces:**
- Consumes: `HybridRetriever.search`.
- Produces: `Citation(label, document_id, filename, location, chunk_id, excerpt)`.
- Produces: `AnswerResult(answer, citations, insufficient_evidence, elapsed_ms)`.
- Produces: `QuestionAnsweringService.answer(question: str) -> AnswerResult`.
- Consumes: `ChatModel.generate(system_prompt, user_prompt) -> str` protocol.

- [ ] **Step 1: Test citation mapping and insufficient-evidence behavior**

Use a fake chat model. Verify `[S1]` resolves to an exact chunk, unknown citation labels are rejected, excerpts are bounded, low retrieval score returns a fixed insufficient-evidence answer without calling the model, and source text containing instructions is quoted as data.

- [ ] **Step 2: Run focused tests and observe failures**

Run: `pytest tests/test_citations.py tests/test_question_answering.py -v`.

- [ ] **Step 3: Implement the grounded prompt and answer validator**

The system prompt must require evidence-only answers, inline source labels and an explicit insufficiency response. It must state that retrieved text cannot alter system instructions. Parse used labels with a strict regular expression and include only cited sources in the response.

- [ ] **Step 4: Verify and commit**

Run: `pytest tests/test_citations.py tests/test_question_answering.py -v`
Run: `pytest -v && ruff check .`.

```bash
git add src/traffic_knowledge/retrieval src/traffic_knowledge/application tests
git commit -m "feat: answer traffic questions with citations"
```

## Task 7: Forecast and Metrics Integrations

**Files:**
- Create: `src/traffic_knowledge/integrations/forecast_client.py`
- Create: `src/traffic_knowledge/integrations/metrics_snapshot.py`
- Create: `tests/fixtures/metrics_snapshot.json`
- Create: `tests/test_forecast_client.py`
- Create: `tests/test_metrics_snapshot.py`

**Interfaces:**
- Produces: `ForecastClient.forecast(model: str, inputs: list) -> ForecastResult`.
- Produces: `MetricsSnapshotRepository.load(path) -> MetricsSnapshot`.
- Raises typed integration errors: `FORECAST_TIMEOUT`, `FORECAST_UNAVAILABLE`, `FORECAST_INVALID_RESPONSE`, `METRICS_SCHEMA_INVALID`.

- [ ] **Step 1: Write mocked HTTP and snapshot schema tests**

Use `httpx.MockTransport`. Cover success, timeout, HTTP 503, malformed output, unsupported schema version, missing dataset/split/horizon and non-finite metrics.

- [ ] **Step 2: Verify tests fail before adapters exist**

Run: `pytest tests/test_forecast_client.py tests/test_metrics_snapshot.py -v`.

- [ ] **Step 3: Implement typed adapters**

The client sends the exact external contract and never imports `traffic-ops-agent` Python modules. Metrics comparisons always retain `dataset`, `split`, `horizon`, `created_at` and environment metadata.

- [ ] **Step 4: Contract-check against the running or test forecasting API**

Start the existing forecasting service only if its checkpoint is available. Bind it to `127.0.0.1` on a configurable unused port. Run one health request and one small forecast request. If unavailable, record the reason; mocked contract tests remain mandatory and sufficient for this task.

- [ ] **Step 5: Verify and commit**

Run: `pytest tests/test_forecast_client.py tests/test_metrics_snapshot.py -v`
Run: `pytest -v && ruff check .`.

```bash
git add src/traffic_knowledge/integrations tests
git commit -m "feat: integrate traffic forecasts and metrics"
```

## Task 8: LangGraph Agent with Three Tools

**Files:**
- Create: `src/traffic_knowledge/domain/agent.py`
- Create: `src/traffic_knowledge/application/agent_graph.py`
- Create: `tests/test_agent_graph.py`

**Interfaces:**
- Consumes: QA service, forecast client and metrics repository.
- Produces: `build_agent_graph(dependencies) -> CompiledStateGraph`.
- Produces: tool functions named exactly `search_traffic_knowledge`, `get_model_metrics`, `run_traffic_forecast`.
- Produces: `AgentResponse(answer, citations, tool_calls, partial, errors)`.

- [ ] **Step 1: Write deterministic routing tests**

Use a fake intent model or explicit structured decisions. Test four routes: knowledge-only, metrics comparison, forecast-only, and combined analysis. Verify maximum tool calls, tool argument validation, forecast failure partial response and no arbitrary code execution tool.

- [ ] **Step 2: Run the focused test and verify failure**

Run: `pytest tests/test_agent_graph.py -v`.

- [ ] **Step 3: Implement explicit graph nodes and bounded transitions**

```text
START -> classify_intent
classify_intent -> knowledge | metrics | forecast | combined
tool node(s) -> compose_grounded_answer
compose_grounded_answer -> END
```

Do not implement an open-ended autonomous loop. Record tool name, sanitized arguments, duration, success and error code for each call.

- [ ] **Step 4: Verify and commit**

Run: `pytest tests/test_agent_graph.py -v`
Run: `pytest -v && ruff check .`.

```bash
git add src/traffic_knowledge/domain src/traffic_knowledge/application tests
git commit -m "feat: orchestrate traffic analysis tools"
```

## Task 9: FastAPI Application

**Files:**
- Create: `src/traffic_knowledge/api/dependencies.py`
- Create: `src/traffic_knowledge/api/app.py`
- Create: `tests/test_api.py`

**Interfaces:**
- Produces endpoints: `GET /health`, `POST /documents`, `GET /documents`, `DELETE /documents/{id}`, `POST /retrieval/search`, `POST /chat`, `GET /benchmarks/latest`.

- [ ] **Step 1: Write API contract tests with dependency overrides**

Cover happy paths, duplicate upload, unsupported type, oversized upload, retrieval validation, cited chat response, partial Agent response, missing benchmark and health dependency states.

- [ ] **Step 2: Run tests and verify missing app failure**

Run: `pytest tests/test_api.py -v`.

- [ ] **Step 3: Implement thin handlers and consistent errors**

Return errors as:

```json
{
  "error": {
    "code": "DOCUMENT_TYPE_UNSUPPORTED",
    "message": "...",
    "details": {}
  }
}
```

Handlers validate HTTP data and delegate. They do not contain chunking, retrieval or Agent business logic.

- [ ] **Step 4: Verify API and commit**

Run: `pytest tests/test_api.py -v`
Run: `pytest -v && ruff check .`
Run: `uvicorn traffic_knowledge.api.app:app --host 127.0.0.1 --port 18100` and verify `/health`, then stop only that process.

```bash
git add src/traffic_knowledge/api tests
git commit -m "feat: expose traffic knowledge API"
```

## Task 10: Streamlit Demonstration UI

**Files:**
- Create: `streamlit_app.py`
- Create: `tests/test_streamlit_app.py`

**Interfaces:**
- Consumes only FastAPI HTTP endpoints; does not import repositories or Agent internals.

- [ ] **Step 1: Write a smoke test for four views and configurable API URL**

Verify the module imports without starting a server and exposes document management, cited QA, forecast/model analysis and benchmark views.

- [ ] **Step 2: Run smoke test and verify failure**

Run: `pytest tests/test_streamlit_app.py -v`.

- [ ] **Step 3: Implement four functional views**

Document management uploads and lists files. QA renders answer and expandable source excerpts. Forecast view collects model and input payload and displays partial errors. Benchmark view renders retrieval and performance comparisons without hard-coded result values.

- [ ] **Step 4: Start services and manually inspect through SSH forwarding**

Run API on `127.0.0.1:18100` and Streamlit on `127.0.0.1:18501`. Forward both ports from the local machine. Verify upload, duplicate message, cited answer and unavailable forecast behavior. Stop only these two project processes.

- [ ] **Step 5: Verify and commit**

Run: `pytest tests/test_streamlit_app.py -v`
Run: `pytest -v && ruff check .`.

```bash
git add streamlit_app.py tests
git commit -m "feat: add traffic knowledge demonstration UI"
```

## Task 11: Evaluation Dataset and Retrieval Metrics

**Files:**
- Create: `evaluation/questions.jsonl`
- Create: `src/traffic_knowledge/evaluation/dataset.py`
- Create: `src/traffic_knowledge/evaluation/retrieval_metrics.py`
- Create: `scripts/evaluate_retrieval.py`
- Create: `tests/test_evaluation_dataset.py`
- Create: `tests/test_retrieval_metrics.py`
- Create: `tests/test_evaluate_retrieval_cli.py`

**Interfaces:**
- Produces: `EvaluationQuestion(id, question, category, expected_answer_points, relevant_chunk_ids, expected_tool)`.
- Produces: `compute_retrieval_metrics(cases) -> RetrievalMetrics` with Hit@1, Hit@3, Recall@k and MRR.
- CLI compares `vector`, `bm25` and `hybrid` on the same corpus and question file.

- [ ] **Step 1: Write schema and metric tests with hand-calculated rankings**

Reject duplicate IDs, fewer than one relevant chunk, unsupported tool names and empty expected points. Verify MRR and Hit@k against a three-case example calculated in the test comments.

- [ ] **Step 2: Verify tests fail before implementation**

Run: `pytest tests/test_evaluation_dataset.py tests/test_retrieval_metrics.py tests/test_evaluate_retrieval_cli.py -v`.

- [ ] **Step 3: Implement validation, metrics and JSON report schema**

Every report includes schema version, Git commit, corpus hash, question-set hash, retrieval settings, runtime environment and per-question rankings.

- [ ] **Step 4: Curate 50 questions after the initial corpus is fixed**

Use at least five categories with at least five questions each: dataset facts, metric interpretation, model mechanisms, model comparison and operational guidance. Each question is manually mapped to one or more actual indexed chunk IDs. Do not generate relevant IDs solely with the system being evaluated. Review every answer point against the source document.

- [ ] **Step 5: Run vector/BM25/hybrid comparison and commit source records**

Run: `python scripts/evaluate_retrieval.py --questions evaluation/questions.jsonl --output artifacts/retrieval_metrics.json`
Expected: valid JSON containing all three strategies and 50 or more cases. `artifacts/` remains ignored.

Run: `pytest -v && ruff check .`.

```bash
git add evaluation/questions.jsonl src/traffic_knowledge/evaluation scripts tests
git commit -m "feat: benchmark traffic knowledge retrieval"
```

## Task 12: Answer, Tool and Resource Benchmarks

**Files:**
- Create: `src/traffic_knowledge/evaluation/answer_metrics.py`
- Create: `src/traffic_knowledge/evaluation/performance.py`
- Create: `scripts/evaluate_rag.py`
- Create: `scripts/run_benchmark.py`
- Create: `tests/test_answer_metrics.py`
- Create: `tests/test_performance_metrics.py`
- Create: `tests/test_benchmark_cli.py`
- Modify: `README.md`

**Interfaces:**
- Produces citation correctness, tool selection accuracy and tool-call success rate.
- Produces optional Ragas faithfulness and answer relevancy for a fixed subset.
- Produces ingestion throughput, query p50/p95, process peak RSS and persisted index bytes.

- [ ] **Step 1: Write deterministic metric and timing tests**

Test percentile calculation, warm-up exclusion, repeated measurement count, RSS sampling lifecycle, citation label/source matching and tool confusion matrix. Use fake clocks and functions where possible.

- [ ] **Step 2: Run focused tests and verify failure**

Run: `pytest tests/test_answer_metrics.py tests/test_performance_metrics.py tests/test_benchmark_cli.py -v`.

- [ ] **Step 3: Implement reproducible benchmark harness**

Record:

```json
{
  "schema_version": "1.0",
  "git_commit": "...",
  "environment": {
    "python": "...",
    "platform": "...",
    "cpu": "...",
    "device": "cpu"
  },
  "configuration": {
    "warmup_runs": 5,
    "measured_runs": 30,
    "top_k": 5
  },
  "metrics": {}
}
```

Measure one request at a time for latency and a separate fixed-concurrency run for throughput. Use `time.perf_counter_ns`. Sample process RSS with `psutil`; report baseline and peak delta. Compute index bytes by recursively summing the project-owned index directory.

- [ ] **Step 4: Run fixed live evaluation**

Use the same frozen corpus and `evaluation/questions.jsonl`. Run deterministic retrieval metrics for all 50+ questions. Run answer metrics on a declared fixed subset. If Ragas requires a paid evaluator that is unavailable, still report citation correctness and tool accuracy and mark LLM-judge metrics `not_run` with a reason; never substitute invented scores.

- [ ] **Step 5: Document commands, limitations and evidence mapping**

README must show exact setup, ingestion, API, Streamlit, evaluation and benchmark commands. Add a table mapping each resume claim to an artifact path and command. Document that the system does not predict real-time Kunming traffic without an authorized local data source.

- [ ] **Step 6: Run final verification and commit**

Run: `pytest -v`
Run: `ruff check .`
Run all four CLIs and validate outputs with `python -m json.tool`.
Run API and Streamlit smoke checks.
Expected: all automated checks pass and benchmark JSON contains environment/configuration metadata.

```bash
git add README.md src/traffic_knowledge/evaluation scripts tests
git commit -m "feat: add reproducible RAG benchmarks"
```

## Task 13: MVP Acceptance, GitHub Publication and Resume Update

**Files:**
- Create: `docs/mvp-acceptance.md`
- Modify: `.env.example` if new safe variables were introduced
- Modify locally after evidence exists: `E:/worksp/STDN-main/paper/resume/周宏_项目完成后最终版简历.md`

**Interfaces:**
- Consumes all prior modules and benchmark artifacts.
- Produces an acceptance record with exact commit hashes and measured numbers.

- [ ] **Step 1: Execute all ten design acceptance checks**

Record pass/fail evidence for PDF/DOCX/Markdown ingestion, duplicate handling, citations, retrieval comparison, three-tool routing, forecast outage, 50 questions, benchmark JSON, non-Docker startup and full test/lint status.

- [ ] **Step 2: Perform a security and repository audit**

Run `git status --short`, `git diff --check`, scan tracked files for API-key patterns, verify runtime data is ignored, and confirm no model weights or uploaded documents are tracked accidentally.

- [ ] **Step 3: Create the GitHub repository and push only after local acceptance**

Create `zouzouqaq/traffic-knowledge-agent`, add SSH origin, push `main`, and verify the remote commit equals local HEAD. Do not force-push.

- [ ] **Step 4: Update the resume with measured evidence only**

Replace future-tense knowledge Agent claims with completed capabilities. Select at most three high-signal results: one retrieval-quality comparison, one latency/resource result and one tool/citation correctness result. Include no number that is absent from a benchmark JSON tied to the accepted commit.

- [ ] **Step 5: Commit the acceptance record**

```bash
git add docs/mvp-acceptance.md .env.example
git commit -m "docs: record traffic knowledge agent MVP acceptance"
git push
```

## Execution Order and Checkpoints

- Checkpoint A after Tasks 1-4: documents can be parsed, chunked, deduplicated and recorded with fake indexing.
- Checkpoint B after Tasks 5-6: real hybrid retrieval and cited answers work locally.
- Checkpoint C after Tasks 7-9: three Agent tools and FastAPI work, including partial failures.
- Checkpoint D after Task 10: browser demonstration is usable.
- Checkpoint E after Tasks 11-12: quality, speed and memory evidence exists.
- Checkpoint F after Task 13: GitHub and resume claims match accepted evidence.

At each checkpoint, report in plain language what now works, which commands passed, what remains, and whether any measured result changes the next task.
