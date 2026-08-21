import numpy as np

from traffic_knowledge.evaluation.runner import build_retrievers
from traffic_knowledge.ingestion.chunking import DocumentChunk


class FakeEmbeddingModel:
    def encode(self, texts):
        return np.asarray(
            [[text.lower().count("gru"), text.lower().count("stdn")] for text in texts],
            dtype=np.float32,
        )


class FakeCollection:
    def __init__(self):
        self.items = {}

    def upsert(self, *, ids, documents, embeddings, metadatas):
        for item in zip(ids, documents, embeddings, metadatas, strict=True):
            item_id, document, embedding, metadata = item
            self.items[item_id] = (document, embedding, metadata)


def _chunk(chunk_id, document_id, text, ordinal):
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        text=text,
        location="Models",
        ordinal=ordinal,
        token_estimate=5,
    )


def test_builds_all_retrievers_from_the_same_supplied_chunks():
    chunks = (
        _chunk("c-gru", "doc-1", "GRU predicts traffic", 0),
        _chunk("c-stdn", "doc-1", "STDN predicts traffic", 1),
    )
    collection = FakeCollection()

    retrievers = build_retrievers(
        chunks=chunks,
        embedding_model=FakeEmbeddingModel(),
        vector_collection=collection,
        filename_resolver=lambda document_id: f"{document_id}.md",
        vector_weight=0.6,
        bm25_weight=0.4,
        rrf_constant=60,
    )

    assert set(retrievers) == {"vector", "bm25", "hybrid"}
    assert set(collection.items) == {"c-gru", "c-stdn"}
    assert retrievers["bm25"].search("GRU", top_k=2)[0].chunk_id == "c-gru"
