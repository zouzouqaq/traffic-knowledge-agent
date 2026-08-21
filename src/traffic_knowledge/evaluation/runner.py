"""Shared construction rules for fair retrieval comparisons."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

from traffic_knowledge.ingestion.chunking import DocumentChunk
from traffic_knowledge.retrieval.bm25 import Bm25Index
from traffic_knowledge.retrieval.hybrid import HybridRetriever
from traffic_knowledge.retrieval.vector import ChromaVectorIndex


def validate_retrieval_configuration(
    *,
    top_k: int,
    vector_weight: float,
    bm25_weight: float,
    rrf_constant: int,
) -> None:
    if top_k < 3:
        raise ValueError("top_k must be at least 3 to compute Hit@3")
    weights = (vector_weight, bm25_weight)
    if (
        any(not math.isfinite(weight) or weight < 0 for weight in weights)
        or sum(weights) <= 0
    ):
        raise ValueError("retrieval weights must be finite, non-negative and not both zero")
    if rrf_constant <= 0:
        raise ValueError("rrf_constant must be greater than zero")


def build_retrievers(
    *,
    chunks: Sequence[DocumentChunk],
    embedding_model,
    vector_collection,
    filename_resolver: Callable[[str], str],
    vector_weight: float,
    bm25_weight: float,
    rrf_constant: int,
) -> dict[str, object]:
    """Build all strategies from exactly the same in-memory corpus snapshot."""

    frozen_chunks = tuple(chunks)
    if not frozen_chunks:
        raise ValueError("evaluation corpus must contain at least one chunk")
    vector = ChromaVectorIndex(
        embedding_model=embedding_model,
        collection=vector_collection,
        filename_resolver=filename_resolver,
    )
    vector.upsert(frozen_chunks)
    bm25 = Bm25Index(filename_resolver=filename_resolver)
    bm25.rebuild(frozen_chunks)
    hybrid = HybridRetriever(
        vector,
        bm25,
        vector_weight=vector_weight,
        bm25_weight=bm25_weight,
        rrf_constant=rrf_constant,
    )
    return {"vector": vector, "bm25": bm25, "hybrid": hybrid}
