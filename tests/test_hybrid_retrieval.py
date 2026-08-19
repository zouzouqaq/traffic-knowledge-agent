import pytest

from traffic_knowledge.domain.retrieval import SearchHit
from traffic_knowledge.retrieval.hybrid import HybridRetriever


def _hit(chunk_id, channel, rank):
    return SearchHit(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        text=f"text-{chunk_id}",
        location="page:1",
        filename=f"{chunk_id}.pdf",
        channels=(channel,),
        ranks=((channel, rank),),
        score=1.0 / rank,
    )


class StaticRetriever:
    def __init__(self, hits):
        self.hits = hits

    def search(self, query, top_k):
        del query
        return tuple(self.hits[:top_k])


def test_hybrid_fuses_duplicate_chunk_once_and_retains_channel_ranks():
    vector = StaticRetriever((_hit("shared", "vector", 1), _hit("v", "vector", 2)))
    bm25 = StaticRetriever((_hit("b", "bm25", 1), _hit("shared", "bm25", 2)))
    retriever = HybridRetriever(vector, bm25, vector_weight=0.6, bm25_weight=0.4)

    hits = retriever.search("traffic", top_k=3)

    assert [hit.chunk_id for hit in hits].count("shared") == 1
    shared = next(hit for hit in hits if hit.chunk_id == "shared")
    assert shared.channels == ("vector", "bm25")
    assert shared.ranks == (("vector", 1), ("bm25", 2))
    assert [hit.chunk_id for hit in hits] == ["shared", "v", "b"]


def test_hybrid_order_is_deterministic_when_scores_tie():
    vector = StaticRetriever((_hit("b", "vector", 1),))
    bm25 = StaticRetriever((_hit("a", "bm25", 1),))
    retriever = HybridRetriever(vector, bm25, vector_weight=0.5, bm25_weight=0.5)

    assert [hit.chunk_id for hit in retriever.search("traffic", 2)] == ["a", "b"]


@pytest.mark.parametrize("top_k", [0, -1])
def test_hybrid_rejects_invalid_top_k(top_k):
    retriever = HybridRetriever(StaticRetriever(()), StaticRetriever(()))

    with pytest.raises(ValueError):
        retriever.search("traffic", top_k)
