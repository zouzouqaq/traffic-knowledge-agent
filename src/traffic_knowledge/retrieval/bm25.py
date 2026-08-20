"""Exact-term lexical retrieval using BM25."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence

from rank_bm25 import BM25Okapi

from traffic_knowledge.domain.retrieval import SearchHit
from traffic_knowledge.ingestion.chunking import DocumentChunk

_TOKEN = re.compile(r"[a-z0-9_+-]+|[\u4e00-\u9fff]", re.IGNORECASE)


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class Bm25Index:
    def __init__(self, filename_resolver: Callable[[str], str] | None = None) -> None:
        self.filename_resolver = filename_resolver or (lambda document_id: document_id)
        self._chunks: tuple[DocumentChunk, ...] = ()
        self._model: BM25Okapi | None = None

    def rebuild(self, chunks: Sequence[DocumentChunk]) -> None:
        self._chunks = tuple(chunks)
        self._model = (
            BM25Okapi([_tokenize(chunk.text) for chunk in self._chunks])
            if self._chunks
            else None
        )

    def search(self, query: str, top_k: int = 5) -> tuple[SearchHit, ...]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        if self._model is None:
            return ()
        scores = self._model.get_scores(_tokenize(query))
        ranked = sorted(
            range(len(self._chunks)),
            key=lambda index: (-float(scores[index]), self._chunks[index].chunk_id),
        )[:top_k]
        hits = []
        for rank, index in enumerate(ranked, start=1):
            chunk = self._chunks[index]
            hits.append(
                SearchHit(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    text=chunk.text,
                    location=chunk.location,
                    filename=self.filename_resolver(chunk.document_id),
                    channels=("bm25",),
                    ranks=(("bm25", rank),),
                    score=float(scores[index]),
                )
            )
        return tuple(hits)
