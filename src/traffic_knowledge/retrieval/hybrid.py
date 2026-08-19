"""Weighted reciprocal-rank fusion for vector and lexical channels."""

from __future__ import annotations

from collections import defaultdict

from traffic_knowledge.domain.retrieval import SearchHit


class HybridRetriever:
    def __init__(
        self,
        vector_retriever,
        bm25_retriever,
        vector_weight: float = 0.6,
        bm25_weight: float = 0.4,
        rrf_constant: int = 60,
    ) -> None:
        if vector_weight < 0 or bm25_weight < 0 or vector_weight + bm25_weight <= 0:
            raise ValueError("retrieval weights must be non-negative and not both zero")
        if rrf_constant <= 0:
            raise ValueError("rrf_constant must be greater than zero")
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.weights = {"vector": vector_weight, "bm25": bm25_weight}
        self.rrf_constant = rrf_constant

    def search(self, query: str, top_k: int = 5) -> tuple[SearchHit, ...]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        vector_hits = self.vector_retriever.search(query, top_k)
        bm25_hits = self.bm25_retriever.search(query, top_k)
        by_id: dict[str, SearchHit] = {}
        scores: defaultdict[str, float] = defaultdict(float)
        channels: defaultdict[str, set[str]] = defaultdict(set)
        ranks: defaultdict[str, dict[str, int]] = defaultdict(dict)
        for channel, hits in (("vector", vector_hits), ("bm25", bm25_hits)):
            for rank, hit in enumerate(hits, start=1):
                by_id.setdefault(hit.chunk_id, hit)
                scores[hit.chunk_id] += self.weights[channel] / (self.rrf_constant + rank)
                channels[hit.chunk_id].add(channel)
                ranks[hit.chunk_id][channel] = rank
        ordered_ids = sorted(by_id, key=lambda chunk_id: (-scores[chunk_id], chunk_id))[:top_k]
        return tuple(
            SearchHit(
                chunk_id=chunk_id,
                document_id=by_id[chunk_id].document_id,
                text=by_id[chunk_id].text,
                location=by_id[chunk_id].location,
                filename=by_id[chunk_id].filename,
                channels=tuple(
                    channel
                    for channel in ("vector", "bm25")
                    if channel in channels[chunk_id]
                ),
                ranks=tuple(
                    (channel, ranks[chunk_id][channel])
                    for channel in ("vector", "bm25")
                    if channel in ranks[chunk_id]
                ),
                score=scores[chunk_id],
            )
            for chunk_id in ordered_ids
        )
