# Traffic Knowledge Agent Design

Date: 2026-08-17
Status: Approved for planning
Repository: `/8t/usr/zhouh2024/projects/traffic-knowledge-agent`

## 1. Purpose

Build a separate, demonstrable traffic-domain RAG and analysis Agent that can:

1. ingest trusted traffic documents;
2. answer domain questions with traceable citations;
3. query model evaluation results from the existing traffic forecasting project;
4. call the existing forecasting API when a user asks for a prediction;
5. produce reproducible quality and performance evidence for a resume and interview.

This repository is separate from `traffic-ops-agent`. The two systems communicate through HTTP and versioned JSON contracts. The knowledge Agent does not train forecasting models and the forecasting project does not implement document RAG.

## 2. MVP Boundary

The MVP is complete only when one end-to-end path works:

```text
traffic document -> duplicate check -> parse -> chunk -> index
-> hybrid retrieval -> cited answer
-> optional forecast/metrics tool call -> analysis response
-> evaluation JSON
```

### Included in MVP

- PDF, DOCX and Markdown ingestion.
- SHA-256 file-level duplicate detection.
- Page or section metadata retained through indexing and retrieval.
- Local Chinese embedding model with Chroma vector storage.
- BM25 keyword retrieval.
- Deterministic hybrid score fusion and top-k retrieval.
- Answers containing source filename, page/section location and supporting excerpt.
- LangGraph workflow with exactly three business tools:
  - `search_traffic_knowledge`
  - `get_model_metrics`
  - `run_traffic_forecast`
- FastAPI endpoints for ingestion, retrieval, chat and health checks.
- Streamlit demonstration interface.
- At least 50 manually verified traffic-domain questions.
- Retrieval, answer quality and resource benchmark reports in JSON.

### Explicitly excluded from MVP

- Vue frontend.
- User registration, roles and permissions.
- Scanned-document OCR.
- Knowledge graph or GraphRAG.
- Multi-agent collaboration.
- Fine-tuning an LLM or embedding model.
- MySQL, Elasticsearch, Redis, MinIO or a mandatory Docker stack.
- Real-time Kunming traffic data integration.
- Automatic traffic-control decisions.

Excluded work may be added only after the MVP acceptance checks pass.

## 3. Architecture

```text
                           +-------------------------+
                           | Streamlit demonstration |
                           +------------+------------+
                                        |
                                        v
+----------------+          +-----------+-----------+
| Traffic files  +--------->| FastAPI application   |
| PDF/DOCX/MD    |          +-----------+-----------+
+----------------+                      |
                              +---------+---------+
                              | LangGraph workflow |
                              +--+--------+-------+
                                 |        |
                    +------------+        +----------------+
                    v                                      v
          +---------+----------+                 +---------+----------+
          | RAG retrieval      |                 | Forecast adapter   |
          | BGE + BM25 + fuse  |                 | HTTP + JSON        |
          +----+-----------+---+                 +---------+----------+
               |           |                               |
               v           v                               v
          +--------+   +---------+              +----------+---------+
          | Chroma |   | SQLite  |              | traffic-ops-agent  |
          | vectors|   | metadata|              | forecast API       |
          +--------+   +---------+              +--------------------+
```

The MVP uses a single Python environment and can run without Docker. SQLite stores document and chunk metadata. Chroma persists vectors locally. Raw documents, indexes and generated artifacts remain outside Git.

## 4. Technology Choices

| Concern | Choice | Reason |
|---|---|---|
| Python runtime | Python 3.11 | Matches the existing server environment |
| API | FastAPI | Existing project experience and typed HTTP contracts |
| Demo UI | Streamlit | Fastest path to a complete, inspectable workflow |
| PDF parsing | PyMuPDF | Retains page boundaries and is lightweight |
| DOCX parsing | python-docx | Extracts paragraphs and headings without a service |
| Embeddings | BAAI `bge-small-zh-v1.5` | Chinese retrieval support and modest resource use |
| Vector store | Chroma | Embedded local persistence; no external daemon |
| Keyword retrieval | BM25 | Interpretable lexical baseline for hybrid evaluation |
| Agent orchestration | LangGraph | Explicit state transitions and testable tool routing |
| LLM access | OpenAI-compatible configurable client | Supports cloud or local endpoints without coupling |
| Evaluation | deterministic IR metrics plus selected Ragas metrics | Separates retrieval quality from generated-answer quality |

Kotaemon is a design reference for hybrid retrieval and citation presentation. No full RAG platform repository is copied. RAGFlow, Dify and FastGPT are not runtime dependencies.

## 5. Repository Structure

```text
traffic-knowledge-agent/
├── pyproject.toml
├── README.md
├── .env.example
├── src/traffic_knowledge/
│   ├── domain/
│   │   ├── document.py
│   │   ├── retrieval.py
│   │   └── agent.py
│   ├── ingestion/
│   │   ├── loaders.py
│   │   ├── chunking.py
│   │   └── service.py
│   ├── retrieval/
│   │   ├── vector.py
│   │   ├── bm25.py
│   │   ├── hybrid.py
│   │   └── citations.py
│   ├── integrations/
│   │   └── forecast_client.py
│   ├── application/
│   │   ├── question_answering.py
│   │   └── agent_graph.py
│   ├── api/
│   │   └── app.py
│   └── evaluation/
│       ├── retrieval_metrics.py
│       ├── answer_metrics.py
│       └── performance.py
├── scripts/
│   ├── ingest_documents.py
│   ├── evaluate_retrieval.py
│   ├── evaluate_rag.py
│   └── run_benchmark.py
├── streamlit_app.py
├── tests/
├── evaluation/questions.jsonl
└── docs/
```

Each module has one responsibility. Framework objects do not enter the domain layer. The forecast client is an adapter so API changes do not affect Agent logic.

## 6. Document Ingestion

### Input

- one PDF, DOCX or Markdown file;
- optional metadata: title, publisher, publication date, category and source URL.

### Processing

1. Validate extension and maximum file size.
2. Stream file bytes through SHA-256; do not load large files twice.
3. Reject a hash already marked as successfully indexed.
4. Parse text while preserving page numbers for PDF and heading paths for DOCX/Markdown.
5. Normalize whitespace without rewriting source meaning.
6. Split text by natural section boundaries, then apply bounded overlap.
7. Assign stable chunk IDs derived from document hash and chunk position.
8. Store document/chunk metadata in SQLite.
9. Create embeddings in batches and persist them in Chroma.
10. Build or refresh the BM25 corpus.

### Output

```json
{
  "document_id": "...",
  "sha256": "...",
  "status": "indexed",
  "chunk_count": 42,
  "elapsed_ms": 815
}
```

If indexing fails, the document is marked `failed`, no success record is returned, and a retry is allowed.

## 7. Retrieval and Citations

The system runs vector and BM25 retrieval independently. Each result list is normalized by rank and combined using a documented weighted reciprocal-rank fusion. The initial weights are configuration, not hidden constants.

Every returned chunk includes:

- document ID and title;
- filename;
- page number or heading path;
- chunk ID;
- retrieval channels and ranks;
- fused score;
- source excerpt.

The generator receives only the selected chunks and must answer under these rules:

1. use retrieved evidence;
2. cite claims as `[S1]`, `[S2]`;
3. state that evidence is insufficient when retrieval is below threshold;
4. never claim that a forecast is observed traffic;
5. keep prediction metrics distinct from RAG quality metrics.

## 8. Agent Workflow

The LangGraph state contains the user question, classified intent, tool calls, retrieved evidence, forecast response, model metrics, answer and errors.

### Tools

`search_traffic_knowledge(query, top_k)` returns cited document chunks.

`get_model_metrics(model_names, dataset)` returns a validated metrics snapshot. During MVP this snapshot is exported from `traffic-ops-agent` as versioned JSON; a later API endpoint can replace the file adapter without changing the tool contract.

`run_traffic_forecast(inputs, model)` calls the existing forecasting HTTP API. The base URL, timeout and retry count are environment configuration.

### Routing

- Domain explanation -> knowledge search.
- Model comparison -> metrics tool plus optional knowledge search.
- Numerical prediction -> forecast tool; knowledge search is used only when explanation is requested.
- Combined analysis -> forecast, metrics and knowledge search, then one grounded report.

The graph has a maximum tool-call count and no unrestricted code execution. Tool errors become structured user-visible partial results instead of fabricated answers.

## 9. Cross-Repository Contract

The knowledge Agent treats `traffic-ops-agent` as an external service.

Required forecast contract:

```json
POST /forecast
{
  "model": "gru",
  "inputs": [[[[0.0]]]]
}
```

The concrete tensor shape remains defined by the forecasting service and is validated before the request.

Metrics are exported as a versioned snapshot:

```json
{
  "schema_version": "1.0",
  "dataset": "PEMS04",
  "created_at": "ISO-8601 timestamp",
  "environment": {},
  "models": []
}
```

The Agent displays the dataset, split, horizon and benchmark timestamp whenever it compares models.

## 10. Evaluation and Resume Evidence

### Forecasting project evidence

The forecasting repository will produce a separate reproducible benchmark report containing:

- MAE, RMSE and MAPE;
- MAE at 5, 15, 30 and 60 minutes;
- warm inference latency p50 and p95;
- throughput in samples per second;
- peak CPU resident memory;
- peak GPU allocated and reserved memory when CUDA is used;
- checkpoint size;
- hardware, software versions, batch size and repetition count.

Warm-up calls are excluded from latency percentiles and recorded separately. GPU timing uses CUDA synchronization. Peak memory counters are reset before each measured run.

### Knowledge Agent evidence

Create at least 50 manually checked questions. Each record contains the question, expected answer points, relevant document/chunk IDs, expected tool and category.

Report:

- vector-only, BM25-only and hybrid Hit@1/Hit@3;
- MRR and Recall@k;
- citation correctness rate;
- tool selection accuracy and tool-call success rate;
- answer faithfulness and answer relevancy on a fixed evaluation subset;
- ingestion documents/minute and chunks/second;
- query latency p50/p95;
- peak process memory;
- persisted index size.

The benchmark compares retrieval strategies on the same corpus and questions. Resume numbers are selected only from committed JSON reports with commands and environment recorded.

No target score is invented in advance. Acceptance requires valid measurement and an interpretable comparison, not a guaranteed flattering result.

## 11. API and UI

Initial endpoints:

- `GET /health`
- `POST /documents`
- `GET /documents`
- `DELETE /documents/{document_id}`
- `POST /retrieval/search`
- `POST /chat`
- `GET /benchmarks/latest`

The Streamlit application contains four functional views:

1. document management;
2. cited question answering;
3. forecast/model analysis;
4. benchmark results.

The UI is an inspection surface, not a marketing page. It must expose evidence and errors rather than hide them.

## 12. Error Handling and Safety

- Unsupported, empty, encrypted or oversized documents are rejected with explicit error codes.
- Duplicate files return the existing document ID.
- Empty parsed content is never indexed.
- Embedding/index writes are idempotent by stable chunk ID.
- LLM and forecast timeouts are bounded.
- API keys exist only in `.env`, which is ignored by Git.
- Uploaded filenames are sanitized; storage paths are generated internally.
- Retrieved document text is treated as untrusted data, not executable instructions.
- The Agent cannot execute arbitrary shell or Python code.
- Deleting a document removes its metadata, BM25 entries and vectors consistently.

## 13. Testing Strategy

The project uses focused tests rather than a red/green interruption for every line. Each coherent batch ends with local tests, full tests and lint checks.

Test layers:

- unit: hashing, parsing, chunking, fusion, citations and metrics;
- contract: forecast client requests/responses and metrics snapshot validation;
- integration: ingest -> retrieve -> cited answer using small local fixtures;
- Agent: deterministic fake LLM verifies routing and tool limits;
- API: FastAPI request/response and error contracts;
- CLI: evaluation scripts write valid JSON;
- smoke: Streamlit starts and the core API health endpoint responds.

Tests must not require a paid API or GPU. Live-model tests are optional and explicitly marked.

## 14. Deployment and Resource Rules

- Work only under `/8t/usr/zhouh2024/projects/traffic-knowledge-agent`.
- Run interactive work in the `tmux b` session using dedicated windows.
- Do not access or modify other users' files.
- Do not use `systemctl`, restart Docker or clean global Docker resources.
- Default to CPU. Check `nvidia-smi` immediately before optional GPU use and use at most one idle GPU.
- Bind development services to `127.0.0.1`; access them through SSH port forwarding.
- Use configurable ports that do not collide with shared services.

## 15. Acceptance Criteria

MVP is accepted when all conditions hold:

1. A PDF, DOCX and Markdown fixture can be indexed.
2. Re-uploading identical bytes does not duplicate chunks.
3. A question returns an answer with a resolvable citation and excerpt.
4. Hybrid retrieval is compared fairly against vector-only and BM25-only retrieval.
5. The Agent selects each of the three tools correctly in deterministic tests.
6. The forecast service being unavailable produces a clear partial response.
7. At least 50 checked evaluation questions exist.
8. Retrieval, answer, latency and memory reports are valid JSON and reproducible from documented commands.
9. FastAPI and Streamlit start without Docker.
10. Full automated tests and lint checks pass.

## 16. Resume Positioning After Acceptance

The two projects remain separate:

- **Traffic forecasting and model evaluation system:** emphasizes data contracts, model training, standardized benchmarking, inference performance and deployment.
- **Traffic-domain knowledge Q&A and analysis Agent:** emphasizes RAG ingestion, hybrid retrieval, citations, LangGraph tool orchestration, API integration and objective RAG evaluation.

Claims about retrieval quality, response latency, memory use or tool accuracy are inserted only after the benchmark artifacts exist. The project does not claim real-time Kunming prediction until an authorized, suitable local data source is integrated.
