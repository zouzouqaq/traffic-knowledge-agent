"""Compare vector, BM25 and hybrid retrieval on one frozen question set."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

from traffic_knowledge.evaluation.dataset import load_evaluation_questions
from traffic_knowledge.evaluation.retrieval_metrics import (
    RetrievalCase,
    build_retrieval_report,
)
from traffic_knowledge.evaluation.runner import (
    build_retrievers,
    validate_retrieval_configuration,
)
from traffic_knowledge.ingestion.repository import DocumentRepository


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--embedding-model",
        default=os.getenv("TRAFFIC_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5"),
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--vector-weight", type=float, default=0.6)
    parser.add_argument("--bm25-weight", type=float, default=0.4)
    parser.add_argument("--rrf-constant", type=int, default=60)
    return parser.parse_args()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _corpus_hash(chunks) -> str:
    digest = hashlib.sha256()
    for chunk in sorted(chunks, key=lambda item: item.chunk_id):
        fields = (
            chunk.chunk_id,
            chunk.document_id,
            str(chunk.ordinal),
            chunk.location,
            chunk.text,
        )
        digest.update("\x1f".join(fields).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _directory_hash(path: Path) -> str:
    if not path.is_dir():
        raise ValueError(
            "embedding model must be a local directory so its files can be fingerprinted"
        )
    digest = hashlib.sha256()
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(file_path.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with file_path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _git_state(project_root: Path) -> tuple[str, bool]:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=True,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip(), bool(status.stdout.strip())


def _runtime_environment() -> dict[str, str]:
    package_versions = {
        package: importlib.metadata.version(package)
        for package in ("chromadb", "sentence-transformers", "rank-bm25", "numpy")
    }
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": platform.processor() or "unknown",
        "device": "cpu",
        **{f"package:{name}": version for name, version in package_versions.items()},
    }


def _cases(questions, retriever, top_k: int) -> tuple[RetrievalCase, ...]:
    return tuple(
        RetrievalCase(
            question_id=question.id,
            relevant_chunk_ids=question.relevant_chunk_ids,
            ranked_chunk_ids=tuple(
                hit.chunk_id for hit in retriever.search(question.question, top_k=top_k)
            ),
        )
        for question in questions
    )


def main() -> None:
    args = parse_args()
    validate_retrieval_configuration(
        top_k=args.top_k,
        vector_weight=args.vector_weight,
        bm25_weight=args.bm25_weight,
        rrf_constant=args.rrf_constant,
    )

    data_dir = args.data_dir.expanduser().resolve()
    repository = DocumentRepository(data_dir / "metadata.sqlite3")
    repository.initialize()
    chunks = repository.list_all_chunks()
    if not chunks:
        raise RuntimeError("the evaluation corpus is empty; ingest documents first")

    questions = load_evaluation_questions(args.questions)
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
        preview = ", ".join(missing_chunk_ids[:5])
        raise ValueError(f"question set references chunks absent from corpus: {preview}")

    import chromadb
    from sentence_transformers import SentenceTransformer

    embedding_model = SentenceTransformer(
        args.embedding_model,
        device="cpu",
        local_files_only=True,
    )

    def filename_resolver(document_id: str) -> str:
        record = repository.find_by_id(document_id)
        return record.filename if record is not None else document_id

    evaluation_client = chromadb.EphemeralClient()
    evaluation_collection = evaluation_client.create_collection(
        name="retrieval_evaluation"
    )
    retrievers = build_retrievers(
        chunks=chunks,
        embedding_model=embedding_model,
        vector_collection=evaluation_collection,
        filename_resolver=filename_resolver,
        vector_weight=args.vector_weight,
        bm25_weight=args.bm25_weight,
        rrf_constant=args.rrf_constant,
    )
    strategy_cases = {
        name: _cases(questions, retriever, args.top_k)
        for name, retriever in retrievers.items()
    }
    project_root = Path(__file__).resolve().parents[1]
    git_commit, git_dirty = _git_state(project_root)
    report = build_retrieval_report(
        strategy_cases=strategy_cases,
        git_commit=git_commit,
        git_dirty=git_dirty,
        corpus_hash=_corpus_hash(chunks),
        question_set_hash=_sha256_file(args.questions),
        retrieval_settings={
            "top_k": args.top_k,
            "vector_weight": args.vector_weight,
            "bm25_weight": args.bm25_weight,
            "rrf_constant": args.rrf_constant,
            "embedding_model": args.embedding_model,
            "embedding_model_hash": _directory_hash(
                Path(args.embedding_model).expanduser().resolve()
            ),
        },
        runtime_environment=_runtime_environment(),
        recall_k=args.top_k,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    json.dump(report, sys.stdout, ensure_ascii=False, indent=2, allow_nan=False)
    sys.stdout.write("\n")
    print(f"Report saved to: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
