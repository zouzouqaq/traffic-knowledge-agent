"""Chroma-backed vector retrieval with an injectable collection for tests."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

from traffic_knowledge.domain.retrieval import SearchHit
from traffic_knowledge.ingestion.chunking import DocumentChunk


class ChromaVectorIndex:
    def __init__(
        self,
        embedding_model,
        chroma_path: Path | None = None,
        collection_name: str = "traffic_knowledge",
        collection=None,
        filename_resolver: Callable[[str], str] | None = None,
    ) -> None:
        self.embedding_model = embedding_model
        self.filename_resolver = filename_resolver or (lambda document_id: document_id)
        if collection is not None:
            self.collection = collection
        else:
            if chroma_path is None:
                raise ValueError("chroma_path is required when collection is not provided")
            import chromadb

            client = chromadb.PersistentClient(path=str(chroma_path))
            self.collection = client.get_or_create_collection(name=collection_name)

    def _encode(self, texts: list[str]) -> np.ndarray:
        embeddings = np.asarray(self.embedding_model.encode(texts), dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[0] != len(texts):
            raise ValueError("embedding model returned an invalid shape")
        return embeddings

    def upsert(self, chunks: tuple[DocumentChunk, ...]) -> None:
        if not chunks:
            return
        embeddings = self._encode([chunk.text for chunk in chunks])
        self.collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            embeddings=embeddings.tolist(),
            metadatas=[
                {
                    "document_id": chunk.document_id,
                    "location": chunk.location,
                    "ordinal": chunk.ordinal,
                }
                for chunk in chunks
            ],
        )

    def search(self, query: str, top_k: int = 5) -> tuple[SearchHit, ...]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        query_embedding = self._encode([query])
        result = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        hits = []
        for rank, (chunk_id, text, metadata, distance) in enumerate(
            zip(ids, documents, metadatas, distances, strict=True), start=1
        ):
            document_id = str(metadata["document_id"])
            hits.append(
                SearchHit(
                    chunk_id=str(chunk_id),
                    document_id=document_id,
                    text=str(text),
                    location=str(metadata.get("location", "")),
                    filename=self.filename_resolver(document_id),
                    channels=("vector",),
                    ranks=(("vector", rank),),
                    score=1.0 / (1.0 + max(float(distance), 0.0)),
                )
            )
        return tuple(hits)

    def delete(self, document_id: str) -> None:
        self.collection.delete(where={"document_id": document_id})
