"""Idempotent document ingestion orchestration."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from traffic_knowledge.domain.document import DocumentValidationError
from traffic_knowledge.ingestion.chunking import DocumentChunk, chunk_document
from traffic_knowledge.ingestion.loaders import load_document
from traffic_knowledge.ingestion.repository import DocumentRepository


class VectorIndex(Protocol):
    def upsert(self, chunks: tuple[DocumentChunk, ...]) -> None: ...


class IngestionError(RuntimeError):
    """An ingestion failure with a stable machine-readable code."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True)
class IngestionResult:
    document_id: str
    sha256: str
    status: str
    chunk_count: int
    elapsed_ms: float
    duplicate: bool


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    """Hash a file incrementally instead of loading it all into memory."""
    if block_size <= 0:
        raise ValueError("block_size must be greater than zero")
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


class IngestionService:
    def __init__(
        self,
        repository: DocumentRepository,
        vector_index: VectorIndex,
        max_file_bytes: int,
        max_characters: int = 1000,
        overlap_characters: int = 100,
    ) -> None:
        self.repository = repository
        self.vector_index = vector_index
        self.max_file_bytes = max_file_bytes
        self.max_characters = max_characters
        self.overlap_characters = overlap_characters

    def ingest(self, path: Path) -> IngestionResult:
        """Hash, parse, chunk, persist and index one source document."""
        started = time.perf_counter()
        path = Path(path)
        if not path.is_file():
            raise DocumentValidationError("DOCUMENT_NOT_FOUND", str(path))
        if path.stat().st_size > self.max_file_bytes:
            raise DocumentValidationError("DOCUMENT_TOO_LARGE", path.name)

        sha256 = sha256_file(path)
        existing = self.repository.find_by_sha256(sha256)
        if existing is not None and existing.status == "indexed":
            return IngestionResult(
                document_id=existing.document_id,
                sha256=sha256,
                status=existing.status,
                chunk_count=len(self.repository.list_chunks(existing.document_id)),
                elapsed_ms=(time.perf_counter() - started) * 1000,
                duplicate=True,
            )

        record = self.repository.begin_ingestion(sha256, path.name)
        try:
            parsed = load_document(path)
            chunks = chunk_document(
                record.document_id,
                parsed,
                self.max_characters,
                self.overlap_characters,
            )
            self.repository.replace_chunks(record.document_id, chunks)
        except DocumentValidationError as error:
            self.repository.mark_failed(record.document_id, error.code)
            raise
        except Exception as error:
            self.repository.mark_failed(record.document_id, "INGESTION_PREPARATION_FAILED")
            raise IngestionError(
                "INGESTION_PREPARATION_FAILED", path.name
            ) from error

        try:
            self.vector_index.upsert(chunks)
        except Exception as error:
            self.repository.mark_failed(record.document_id, "VECTOR_INDEX_FAILED")
            raise IngestionError("VECTOR_INDEX_FAILED", path.name) from error

        self.repository.mark_indexed(record.document_id)
        return IngestionResult(
            document_id=record.document_id,
            sha256=sha256,
            status="indexed",
            chunk_count=len(chunks),
            elapsed_ms=(time.perf_counter() - started) * 1000,
            duplicate=False,
        )
