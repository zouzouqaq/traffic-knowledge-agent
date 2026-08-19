import pytest

from traffic_knowledge.settings import Settings


def test_settings_use_hybrid_retrieval_defaults(monkeypatch):
    monkeypatch.delenv("TRAFFIC_RETRIEVAL_VECTOR_WEIGHT", raising=False)
    monkeypatch.delenv("TRAFFIC_RETRIEVAL_BM25_WEIGHT", raising=False)
    monkeypatch.delenv("TRAFFIC_EMBEDDING_MODEL", raising=False)

    settings = Settings.from_env()

    assert settings.retrieval_vector_weight == 0.6
    assert settings.retrieval_bm25_weight == 0.4
    assert settings.embedding_model_name == "BAAI/bge-small-zh-v1.5"


def test_settings_reject_both_retrieval_weights_as_zero(monkeypatch):
    monkeypatch.setenv("TRAFFIC_RETRIEVAL_VECTOR_WEIGHT", "0")
    monkeypatch.setenv("TRAFFIC_RETRIEVAL_BM25_WEIGHT", "0")

    with pytest.raises(ValueError):
        Settings.from_env()
