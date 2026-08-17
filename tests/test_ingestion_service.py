import hashlib

import pytest

from traffic_knowledge.domain.document import DocumentValidationError
from traffic_knowledge.ingestion.repository import DocumentRepository
from traffic_knowledge.ingestion.service import (
    IngestionError,
    IngestionService,
    sha256_file,
)


class FakeVectorIndex:
    def __init__(self, failures: int = 0) -> None:
        self.failures = failures
        self.upserted = []

    def upsert(self, chunks) -> None:
        if self.failures:
            self.failures -= 1
            raise RuntimeError("index unavailable")
        self.upserted.extend(chunks)


def _service(tmp_path, index=None, max_file_bytes=1024):
    repository = DocumentRepository(tmp_path / "metadata.sqlite3")
    repository.initialize()
    return (
        IngestionService(
            repository=repository,
            vector_index=index or FakeVectorIndex(),
            max_file_bytes=max_file_bytes,
            max_characters=40,
            overlap_characters=5,
        ),
        repository,
    )


def test_ingests_document_and_detects_identical_bytes(tmp_path):
    path = tmp_path / "guide.md"
    path.write_text("# Traffic\nGRU predicts traffic flow.", encoding="utf-8")
    index = FakeVectorIndex()
    service, repository = _service(tmp_path, index)

    first = service.ingest(path)
    duplicate = service.ingest(path)

    assert first.status == "indexed"
    assert first.duplicate is False
    assert first.chunk_count == 1
    assert duplicate.document_id == first.document_id
    assert duplicate.duplicate is True
    assert len(index.upserted) == 1
    assert repository.find_by_sha256(first.sha256).status == "indexed"


def test_same_filename_with_different_bytes_creates_new_document(tmp_path):
    path = tmp_path / "guide.md"
    service, _ = _service(tmp_path)
    path.write_text("# Traffic\nFirst version.", encoding="utf-8")
    first = service.ingest(path)
    path.write_text("# Traffic\nSecond version.", encoding="utf-8")

    second = service.ingest(path)

    assert second.document_id != first.document_id
    assert second.sha256 != first.sha256


def test_rejects_file_over_size_limit_before_metadata_write(tmp_path):
    path = tmp_path / "large.md"
    path.write_text("# Traffic\n" + "x" * 100, encoding="utf-8")
    service, repository = _service(tmp_path, max_file_bytes=20)

    with pytest.raises(DocumentValidationError) as error:
        service.ingest(path)

    assert error.value.code == "DOCUMENT_TOO_LARGE"
    assert repository.find_by_sha256(sha256_file(path)) is None


def test_index_failure_is_recorded_and_can_be_retried(tmp_path):
    path = tmp_path / "guide.md"
    path.write_text("# Traffic\nRetry this document.", encoding="utf-8")
    index = FakeVectorIndex(failures=1)
    service, repository = _service(tmp_path, index)

    with pytest.raises(IngestionError) as error:
        service.ingest(path)
    assert error.value.code == "VECTOR_INDEX_FAILED"

    failed = repository.find_by_sha256(sha256_file(path))
    assert failed.status == "failed"
    assert failed.error_code == "VECTOR_INDEX_FAILED"

    retried = service.ingest(path)
    assert retried.document_id == failed.document_id
    assert retried.status == "indexed"
    assert retried.duplicate is False


def test_sha256_file_matches_hashlib_for_multiple_read_blocks(tmp_path):
    path = tmp_path / "payload.md"
    payload = b"traffic-flow-data" * 100
    path.write_bytes(payload)

    assert sha256_file(path, block_size=7) == hashlib.sha256(payload).hexdigest()
