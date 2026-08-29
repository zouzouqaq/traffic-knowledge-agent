"""Run deterministic Agent quality, latency and resource benchmarks."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import threading
import time
from pathlib import Path

from traffic_knowledge.api.dependencies import EvidenceOnlyChatModel, RuleBasedIntentModel
from traffic_knowledge.application.agent_graph import AgentDependencies, build_agent_graph
from traffic_knowledge.application.question_answering import QuestionAnsweringService
from traffic_knowledge.evaluation.answer_metrics import (
    AnswerEvaluationCase,
    build_benchmark_agent_input,
    classify_tool_calls,
    compute_answer_metrics,
)
from traffic_knowledge.evaluation.dataset import load_evaluation_questions
from traffic_knowledge.evaluation.performance import (
    benchmark_callable,
    benchmark_throughput,
    directory_size_bytes,
)
from traffic_knowledge.evaluation.provenance import (
    corpus_sha256,
    directory_sha256,
    file_sha256,
    git_state,
    runtime_environment,
)
from traffic_knowledge.ingestion.repository import DocumentRepository
from traffic_knowledge.ingestion.service import IngestionService
from traffic_knowledge.integrations.metrics_snapshot import MetricsSnapshotRepository
from traffic_knowledge.retrieval.bm25 import Bm25Index
from traffic_knowledge.retrieval.hybrid import HybridRetriever
from traffic_knowledge.retrieval.vector import ChromaVectorIndex


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics-path", type=Path, required=True)
    parser.add_argument("--embedding-model", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, default=Path("artifacts/benchmark_index"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--warmup-runs", type=int, default=5)
    parser.add_argument("--measured-runs", type=int, default=30)
    parser.add_argument("--throughput-concurrency", type=int, default=4)
    parser.add_argument("--expected-question-count", type=int, default=50)
    return parser.parse_args()


class _UnusedForecastClient:
    def forecast(self, model, inputs):
        del model, inputs
        raise RuntimeError("forecast questions are outside this fixed benchmark")


def main() -> None:
    args = parse_args()
    if (
        args.top_k <= 0
        or args.warmup_runs < 0
        or args.measured_runs <= 0
        or args.throughput_concurrency <= 0
        or args.throughput_concurrency > args.measured_runs
        or args.expected_question_count <= 0
    ):
        raise ValueError("benchmark run counts and top_k are invalid")
    questions = load_evaluation_questions(args.questions)
    if len(questions) != args.expected_question_count:
        raise ValueError(
            f"expected {args.expected_question_count} questions, got {len(questions)}"
        )

    import chromadb
    import psutil
    from sentence_transformers import SentenceTransformer

    index_dir = args.index_dir.expanduser().resolve()
    if index_dir.name != "benchmark_index":
        raise ValueError("index-dir must end with benchmark_index")
    shutil.rmtree(index_dir, ignore_errors=True)
    index_dir.mkdir(parents=True)
    embedding_model = SentenceTransformer(
        str(args.embedding_model), device="cpu", local_files_only=True
    )
    client = chromadb.PersistentClient(path=str(index_dir))
    collection = client.create_collection("traffic_benchmark")
    repository = DocumentRepository(index_dir / "metadata.sqlite3")
    repository.initialize()

    def filename_resolver(document_id: str) -> str:
        record = repository.find_by_id(document_id)
        return record.filename if record is not None else document_id

    vector_index = ChromaVectorIndex(
        embedding_model=embedding_model,
        collection=collection,
        filename_resolver=filename_resolver,
    )
    ingestion_service = IngestionService(
        repository=repository,
        vector_index=vector_index,
        max_file_bytes=50 * 1024 * 1024,
    )
    corpus_dir = args.corpus_dir.expanduser().resolve()
    source_files = sorted(
        path
        for path in corpus_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".md", ".pdf", ".docx"}
    )
    if not source_files:
        raise RuntimeError("benchmark corpus directory has no supported documents")
    ingestion_started = time.perf_counter_ns()
    ingestion_results = tuple(
        ingestion_service.ingest(path) for path in source_files
    )
    ingestion_seconds = (time.perf_counter_ns() - ingestion_started) / 1_000_000_000
    chunks = repository.list_all_chunks()
    indexed_chunk_ids = {chunk.chunk_id for chunk in chunks}
    missing_chunk_ids = sorted(
        {
            chunk_id
            for question in questions
            for chunk_id in question.relevant_chunk_ids
            if chunk_id not in indexed_chunk_ids
        }
    )
    if missing_chunk_ids:
        raise ValueError(
            "question set references chunks absent from corpus: "
            + ", ".join(missing_chunk_ids[:5])
        )
    bm25_index = Bm25Index(filename_resolver=filename_resolver)
    bm25_index.rebuild(chunks)
    hybrid_retriever = HybridRetriever(
        vector_index,
        bm25_index,
        vector_weight=0.6,
        bm25_weight=0.4,
        rrf_constant=60,
    )
    qa_service = QuestionAnsweringService(
        retriever=hybrid_retriever,
        chat_model=EvidenceOnlyChatModel(),
        top_k=args.top_k,
    )
    graph = build_agent_graph(
        AgentDependencies(
            intent_model=RuleBasedIntentModel(),
            qa_service=qa_service,
            forecast_client=_UnusedForecastClient(),
            metrics_repository=MetricsSnapshotRepository(),
            metrics_path=args.metrics_path,
        )
    )

    cases = []
    per_question = []
    for question in questions:
        response = graph.invoke(build_benchmark_agent_input(question.question))["response"]
        selected_tool = classify_tool_calls(
            tuple(call.name for call in response.tool_calls)
        )
        success = bool(response.tool_calls) and all(
            call.success for call in response.tool_calls
        )
        cited_chunk_ids = tuple(citation.chunk_id for citation in response.citations)
        cases.append(
            AnswerEvaluationCase(
                question_id=question.id,
                expected_tool=question.expected_tool,
                selected_tool=selected_tool,
                relevant_chunk_ids=question.relevant_chunk_ids,
                cited_chunk_ids=cited_chunk_ids,
                tool_call_success=success,
            )
        )
        per_question.append(
            {
                "question_id": question.id,
                "expected_tool": question.expected_tool,
                "selected_tool": selected_tool,
                "relevant_chunk_ids": list(question.relevant_chunk_ids),
                "cited_chunk_ids": list(cited_chunk_ids),
                "tool_call_success": success,
            }
        )

    knowledge_questions = [q for q in questions if q.expected_tool == "knowledge"]
    query_index = 0
    query_lock = threading.Lock()

    def invoke_next_question():
        nonlocal query_index
        with query_lock:
            question = knowledge_questions[query_index % len(knowledge_questions)]
            query_index += 1
        return graph.invoke(build_benchmark_agent_input(question.question))

    process = psutil.Process()
    performance = benchmark_callable(
        invoke_next_question,
        warmup_runs=args.warmup_runs,
        measured_runs=args.measured_runs,
        rss_bytes=lambda: process.memory_info().rss,
    )
    throughput = benchmark_throughput(
        invoke_next_question,
        request_count=args.measured_runs,
        concurrency=args.throughput_concurrency,
    )
    project_root = Path(__file__).resolve().parents[1]
    commit, dirty, working_tree_hash = git_state(project_root)
    report = {
        "schema_version": "1.0",
        "git_commit": commit,
        "git_dirty": dirty,
        "git_working_tree_hash": working_tree_hash,
        "environment": runtime_environment(
            ("chromadb", "sentence-transformers", "rank-bm25", "numpy", "psutil")
        ),
        "input_fingerprints": {
            "question_set_sha256": file_sha256(args.questions),
            "corpus_directory_sha256": directory_sha256(corpus_dir),
            "corpus_chunks_sha256": corpus_sha256(chunks),
            "embedding_model_sha256": directory_sha256(args.embedding_model),
            "metrics_snapshot_sha256": file_sha256(args.metrics_path),
        },
        "configuration": {
            "warmup_runs": args.warmup_runs,
            "measured_runs": args.measured_runs,
            "throughput_concurrency": args.throughput_concurrency,
            "top_k": args.top_k,
            "question_count": len(questions),
            "embedding_model": str(args.embedding_model),
            "vector_weight": 0.6,
            "bm25_weight": 0.4,
            "rrf_constant": 60,
        },
        "quality": compute_answer_metrics(tuple(cases)).to_dict(),
        "performance": {
            **performance.to_dict(),
            "concurrent_throughput": throughput.to_dict(),
        },
        "ingestion": {
            "document_count": len(ingestion_results),
            "chunk_count": len(chunks),
            "elapsed_seconds": ingestion_seconds,
            "chunks_per_second": len(chunks) / ingestion_seconds,
            "persisted_index_bytes": directory_size_bytes(index_dir),
        },
        "llm_judge_metrics": {
            "status": "not_run",
            "reason": "No independent evaluator model was configured for this run.",
        },
        "per_question": per_question,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    args.output.write_text(serialized, encoding="utf-8")
    sys.stdout.write(serialized)
    print(f"Report saved to: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
