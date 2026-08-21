"""SQLite metadata repository with explicit transactional writes."""

from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from traffic_knowledge.ingestion.chunking import DocumentChunk


@dataclass(frozen=True)
class DocumentRecord:
    document_id: str
    sha256: str
    filename: str
    status: str
    error_code: str | None
    created_at: str
    updated_at: str


def _now() -> str:
    return datetime.now(UTC).isoformat()


class DocumentRepository:
    """Persist document states and chunk metadata in one local SQLite file."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = Path(database_path)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                  document_id TEXT PRIMARY KEY,
                  sha256 TEXT NOT NULL UNIQUE,
                  filename TEXT NOT NULL,
                  status TEXT NOT NULL
                    CHECK(status IN ('indexing','indexed','failed')),
                  error_code TEXT,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS chunks (
                  chunk_id TEXT PRIMARY KEY,
                  document_id TEXT NOT NULL
                    REFERENCES documents(document_id) ON DELETE CASCADE,
                  ordinal INTEGER NOT NULL,
                  location TEXT NOT NULL,
                  text TEXT NOT NULL,
                  token_estimate INTEGER NOT NULL,
                  UNIQUE(document_id, ordinal)
                );
                """
            )

    @staticmethod
    def _record(row: sqlite3.Row) -> DocumentRecord:
        return DocumentRecord(
            document_id=row["document_id"],
            sha256=row["sha256"],
            filename=row["filename"],
            status=row["status"],
            error_code=row["error_code"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def find_by_sha256(self, sha256: str) -> DocumentRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE sha256 = ?", (sha256,)
            ).fetchone()
        return self._record(row) if row is not None else None

    def find_by_id(self, document_id: str) -> DocumentRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE document_id = ?", (document_id,)
            ).fetchone()
        return self._record(row) if row is not None else None

    def list_documents(self) -> tuple[DocumentRecord, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM documents ORDER BY created_at, document_id"
            ).fetchall()
        return tuple(self._record(row) for row in rows)

    def begin_ingestion(self, sha256: str, filename: str) -> DocumentRecord:
        timestamp = _now()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE sha256 = ?", (sha256,)
            ).fetchone()
            if row is None:
                document_id = uuid.uuid5(
                    uuid.NAMESPACE_URL, f"traffic-knowledge:{sha256}"
                ).hex
                connection.execute(
                    """
                    INSERT INTO documents (
                      document_id, sha256, filename, status, error_code,
                      created_at, updated_at
                    ) VALUES (?, ?, ?, 'indexing', NULL, ?, ?)
                    """,
                    (document_id, sha256, filename, timestamp, timestamp),
                )
            else:
                document_id = row["document_id"]
                connection.execute(
                    """
                    UPDATE documents
                    SET filename = ?, status = 'indexing', error_code = NULL,
                        updated_at = ?
                    WHERE document_id = ?
                    """,
                    (filename, timestamp, document_id),
                )
            stored = connection.execute(
                "SELECT * FROM documents WHERE document_id = ?", (document_id,)
            ).fetchone()
        if stored is None:
            raise RuntimeError("document write did not return a record")
        return self._record(stored)

    def replace_chunks(
        self, document_id: str, chunks: tuple[DocumentChunk, ...]
    ) -> None:
        if any(chunk.document_id != document_id for chunk in chunks):
            raise ValueError("every chunk must belong to document_id")

        with self._connect() as connection:
            connection.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
            connection.executemany(
                """
                INSERT INTO chunks (
                  chunk_id, document_id, ordinal, location, text, token_estimate
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    (
                        chunk.chunk_id,
                        chunk.document_id,
                        chunk.ordinal,
                        chunk.location,
                        chunk.text,
                        chunk.token_estimate,
                    )
                    for chunk in chunks
                ),
            )

    def list_chunks(self, document_id: str) -> tuple[DocumentChunk, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM chunks WHERE document_id = ? ORDER BY ordinal",
                (document_id,),
            ).fetchall()
        return tuple(
            DocumentChunk(
                chunk_id=row["chunk_id"],
                document_id=row["document_id"],
                text=row["text"],
                location=row["location"],
                ordinal=row["ordinal"],
                token_estimate=row["token_estimate"],
            )
            for row in rows
        )

    def list_all_chunks(self) -> tuple[DocumentChunk, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT chunks.* FROM chunks
                JOIN documents USING (document_id)
                WHERE documents.status IN ('indexing', 'indexed')
                ORDER BY documents.created_at, chunks.ordinal, chunks.chunk_id
                """
            ).fetchall()
        return tuple(
            DocumentChunk(
                chunk_id=row["chunk_id"],
                document_id=row["document_id"],
                text=row["text"],
                location=row["location"],
                ordinal=row["ordinal"],
                token_estimate=row["token_estimate"],
            )
            for row in rows
        )

    def _set_status(
        self, document_id: str, status: str, error_code: str | None
    ) -> None:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE documents
                SET status = ?, error_code = ?, updated_at = ?
                WHERE document_id = ?
                """,
                (status, error_code, _now(), document_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(document_id)

    def mark_indexed(self, document_id: str) -> None:
        self._set_status(document_id, "indexed", None)

    def mark_failed(self, document_id: str, error_code: str) -> None:
        self._set_status(document_id, "failed", error_code)

    def delete(self, document_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM documents WHERE document_id = ?", (document_id,)
            )
        return cursor.rowcount == 1
