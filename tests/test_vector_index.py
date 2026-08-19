import numpy as np

from traffic_knowledge.ingestion.chunking import DocumentChunk
from traffic_knowledge.retrieval.vector import ChromaVectorIndex


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
            self.items[item_id] = (document, np.asarray(embedding), metadata)

    def query(self, *, query_embeddings, n_results, include):
        del include
        query = np.asarray(query_embeddings[0])
        ranked = sorted(
            self.items.items(),
            key=lambda item: (
                float(np.square(item[1][1] - query).sum()),
                item[0],
            ),
        )[:n_results]
        return {
            "ids": [[item_id for item_id, _ in ranked]],
            "documents": [[value[0] for _, value in ranked]],
            "metadatas": [[value[2] for _, value in ranked]],
            "distances": [
                [float(np.square(value[1] - query).sum()) for _, value in ranked]
            ],
        }

    def delete(self, *, where):
        self.items = {
            item_id: value
            for item_id, value in self.items.items()
            if value[2]["document_id"] != where["document_id"]
        }


def _chunk(chunk_id, document_id, text, ordinal=0):
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        text=text,
        location="Traffic > Models",
        ordinal=ordinal,
        token_estimate=5,
    )


def test_vector_index_upserts_searches_and_deletes_document():
    collection = FakeCollection()
    index = ChromaVectorIndex(
        embedding_model=FakeEmbeddingModel(),
        collection=collection,
        filename_resolver=lambda document_id: f"{document_id}.md",
    )
    index.upsert(
        (
            _chunk("c-gru", "doc-gru", "GRU GRU predicts traffic"),
            _chunk("c-stdn", "doc-stdn", "STDN STDN predicts traffic"),
        )
    )

    hits = index.search("GRU", top_k=2)

    assert [hit.chunk_id for hit in hits] == ["c-gru", "c-stdn"]
    assert hits[0].filename == "doc-gru.md"
    assert hits[0].channels == ("vector",)
    assert hits[0].ranks == (("vector", 1),)

    index.delete("doc-gru")
    remaining = index.search("GRU", top_k=2)
    assert [hit.chunk_id for hit in remaining] == ["c-stdn"]
