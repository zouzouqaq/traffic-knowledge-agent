import pytest

from traffic_knowledge.ingestion.chunking import DocumentChunk
from traffic_knowledge.retrieval.bm25 import Bm25Index


def test_empty_index_can_be_rebuilt_and_searched():
    index = Bm25Index()

    index.rebuild(())

    assert index.search("traffic", top_k=5) == ()


def _chunk(chunk_id, document_id, text, ordinal=0):
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        text=text,
        location="Traffic > Models",
        ordinal=ordinal,
        token_estimate=5,
    )


def test_bm25_favors_exact_technical_term_and_is_deterministic():
    index = Bm25Index(filename_resolver=lambda document_id: f"{document_id}.md")
    index.rebuild(
        (
            _chunk("c2", "doc-2", "GRU recurrent traffic forecasting"),
            _chunk("c1", "doc-1", "STDN spatial temporal decoupling"),
            _chunk("c3", "doc-3", "Traffic forecasting baseline"),
        )
    )

    first = index.search("STDN", top_k=3)
    second = index.search("STDN", top_k=3)

    assert first[0].chunk_id == "c1"
    assert first == second
    assert first[0].channels == ("bm25",)


def test_bm25_rejects_invalid_top_k():
    index = Bm25Index()

    with pytest.raises(ValueError):
        index.search("traffic", top_k=0)
