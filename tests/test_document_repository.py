import sqlite3

import pytest

from traffic_knowledge.ingestion.chunking import DocumentChunk
from traffic_knowledge.ingestion.repository import DocumentRepository


def _chunk(
    chunk_id: str,
    document_id: str,
    ordinal: int,
    text: str = "Traffic flow",
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        text=text,
        location="page:1",
        ordinal=ordinal,
        token_estimate=3,
    )


def test_records_document_chunks_and_indexed_status(tmp_path):
    repository = DocumentRepository(tmp_path / "metadata.sqlite3")
    repository.initialize()

    record = repository.begin_ingestion("abc123", "guide.pdf")
    repository.replace_chunks(record.document_id, (_chunk("c1", record.document_id, 0),))
    repository.mark_indexed(record.document_id)

    stored = repository.find_by_sha256("abc123")
    assert stored is not None
    assert stored.status == "indexed"
    assert stored.error_code is None
    assert repository.list_chunks(record.document_id)[0].chunk_id == "c1"


def test_chunk_replacement_rolls_back_on_failure(tmp_path):
    repository = DocumentRepository(tmp_path / "metadata.sqlite3")
    repository.initialize()
    record = repository.begin_ingestion("abc123", "guide.pdf")
    original = _chunk("original", record.document_id, 0, "Original text")
    repository.replace_chunks(record.document_id, (original,))

    duplicate_ordinals = (
        _chunk("new-1", record.document_id, 0),
        _chunk("new-2", record.document_id, 0),
    )
    with pytest.raises(sqlite3.IntegrityError):
        repository.replace_chunks(record.document_id, duplicate_ordinals)

    assert repository.list_chunks(record.document_id) == (original,)


def test_failed_ingestion_can_retry_with_same_document_id(tmp_path):
    repository = DocumentRepository(tmp_path / "metadata.sqlite3")
    repository.initialize()
    first = repository.begin_ingestion("abc123", "guide.pdf")
    repository.mark_failed(first.document_id, "VECTOR_INDEX_FAILED")

    failed = repository.find_by_sha256("abc123")
    assert failed is not None
    assert failed.status == "failed"
    assert failed.error_code == "VECTOR_INDEX_FAILED"

    retry = repository.begin_ingestion("abc123", "guide.pdf")
    assert retry.document_id == first.document_id
    assert retry.status == "indexing"
    assert retry.error_code is None


def test_delete_cascades_to_chunks(tmp_path):
    repository = DocumentRepository(tmp_path / "metadata.sqlite3")
    repository.initialize()
    record = repository.begin_ingestion("abc123", "guide.pdf")
    repository.replace_chunks(record.document_id, (_chunk("c1", record.document_id, 0),))

    assert repository.delete(record.document_id) is True
    assert repository.list_chunks(record.document_id) == ()
    assert repository.find_by_sha256("abc123") is None
