import pytest

from traffic_knowledge.evaluation.retrieval_metrics import (
    RetrievalCase,
    build_retrieval_report,
    compute_retrieval_metrics,
)


def test_computes_hand_calculated_retrieval_metrics():
    # q1: RR=1, recall=1; q2: RR=1/2, recall=1/2; q3: RR=0, recall=0.
    # Means: Hit@1=1/3, Hit@3=2/3, Recall@3=1/2, MRR=1/2.
    cases = (
        RetrievalCase("q1", ("a",), ("a", "x", "y")),
        RetrievalCase("q2", ("b", "c"), ("x", "b", "y")),
        RetrievalCase("q3", ("d",), ("x", "y", "z")),
    )

    metrics = compute_retrieval_metrics(cases, recall_k=3)

    assert metrics.case_count == 3
    assert metrics.hit_at_1 == pytest.approx(1 / 3)
    assert metrics.hit_at_3 == pytest.approx(2 / 3)
    assert metrics.recall_at_k == pytest.approx(1 / 2)
    assert metrics.recall_k == 3
    assert metrics.mrr == pytest.approx(1 / 2)


def test_deduplicates_retrieved_ids_before_ranking():
    case = RetrievalCase("q1", ("relevant",), ("noise", "noise", "relevant"))

    metrics = compute_retrieval_metrics((case,), recall_k=3)

    assert metrics.mrr == pytest.approx(1 / 2)


def test_rejects_empty_cases_and_invalid_recall_k():
    with pytest.raises(ValueError, match="at least one"):
        compute_retrieval_metrics(())

    with pytest.raises(ValueError, match="recall_k"):
        compute_retrieval_metrics(
            (RetrievalCase("q1", ("a",), ("a",)),),
            recall_k=0,
        )


def test_rejects_case_without_relevant_chunks():
    with pytest.raises(ValueError, match="relevant_chunk_ids"):
        RetrievalCase("q1", (), ("a",))


def test_builds_reproducible_report_with_per_question_rankings():
    cases = (RetrievalCase("q1", ("a",), ("a", "b")),)

    report = build_retrieval_report(
        strategy_cases={"vector": cases, "bm25": cases, "hybrid": cases},
        git_commit="abc123",
        git_dirty=True,
        corpus_hash="corpus-sha256",
        question_set_hash="questions-sha256",
        retrieval_settings={"top_k": 2, "vector_weight": 0.6},
        runtime_environment={"python": "3.11", "platform": "linux", "device": "cpu"},
        recall_k=2,
    )

    assert report["schema_version"] == "1.0"
    assert report["git_commit"] == "abc123"
    assert report["git_dirty"] is True
    assert report["corpus_hash"] == "corpus-sha256"
    assert report["question_set_hash"] == "questions-sha256"
    assert report["retrieval_settings"]["top_k"] == 2
    assert report["runtime_environment"]["device"] == "cpu"
    assert set(report["strategies"]) == {"vector", "bm25", "hybrid"}
    assert report["strategies"]["hybrid"]["metrics"]["hit_at_1"] == 1.0
    assert report["strategies"]["hybrid"]["rankings"] == [
        {
            "question_id": "q1",
            "relevant_chunk_ids": ["a"],
            "ranked_chunk_ids": ["a", "b"],
        }
    ]


def test_rejects_strategy_results_for_different_question_sets():
    vector_cases = (RetrievalCase("q1", ("a",), ("a",)),)
    bm25_cases = (RetrievalCase("q2", ("b",), ("b",)),)

    with pytest.raises(ValueError, match="same questions"):
        build_retrieval_report(
            strategy_cases={"vector": vector_cases, "bm25": bm25_cases},
            git_commit="abc123",
            git_dirty=False,
            corpus_hash="corpus",
            question_set_hash="questions",
            retrieval_settings={"top_k": 5},
            runtime_environment={"python": "3.11"},
            recall_k=5,
        )


def test_rejects_different_relevance_judgments_for_same_question_id():
    vector_cases = (RetrievalCase("q1", ("a",), ("a",)),)
    bm25_cases = (RetrievalCase("q1", ("b",), ("b",)),)

    with pytest.raises(ValueError, match="same questions"):
        build_retrieval_report(
            strategy_cases={"vector": vector_cases, "bm25": bm25_cases},
            git_commit="abc123",
            git_dirty=False,
            corpus_hash="corpus",
            question_set_hash="questions",
            retrieval_settings={"top_k": 5},
            runtime_environment={"python": "3.11"},
            recall_k=5,
        )


def test_rejects_duplicate_question_ids_within_a_strategy():
    duplicate_cases = (
        RetrievalCase("q1", ("a",), ("a",)),
        RetrievalCase("q1", ("a",), ("a",)),
    )

    with pytest.raises(ValueError, match="duplicate question id"):
        build_retrieval_report(
            strategy_cases={"vector": duplicate_cases},
            git_commit="abc123",
            git_dirty=False,
            corpus_hash="corpus",
            question_set_hash="questions",
            retrieval_settings={"top_k": 5},
            runtime_environment={"python": "3.11"},
            recall_k=5,
        )
