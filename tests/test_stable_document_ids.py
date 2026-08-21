import sqlite3

from traffic_knowledge.ingestion.repository import DocumentRepository


def test_same_sha256_has_same_document_id_across_fresh_databases(tmp_path):
    first_repository = DocumentRepository(tmp_path / "first.sqlite3")
    second_repository = DocumentRepository(tmp_path / "second.sqlite3")
    first_repository.initialize()
    second_repository.initialize()

    first = first_repository.begin_ingestion("a" * 64, "traffic.md")
    second = second_repository.begin_ingestion("a" * 64, "renamed.md")

    assert first.document_id == second.document_id == "207b1eb8b5855184a2fe8b28727f4950"


def test_existing_legacy_document_id_is_preserved_on_retry(tmp_path):
    repository = DocumentRepository(tmp_path / "metadata.sqlite3")
    repository.initialize()
    with sqlite3.connect(repository.database_path) as connection:
        connection.execute(
            """
            INSERT INTO documents (
              document_id, sha256, filename, status, error_code, created_at, updated_at
            ) VALUES (?, ?, ?, 'failed', ?, ?, ?)
            """,
            ("legacy-random-id", "b" * 64, "old.md", "FAILED", "before", "before"),
        )

    retried = repository.begin_ingestion("b" * 64, "old.md")

    assert retried.document_id == "legacy-random-id"
