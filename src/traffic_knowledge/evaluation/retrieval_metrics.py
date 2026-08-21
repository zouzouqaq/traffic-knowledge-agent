"""Deterministic ranking metrics for retrieval evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


def _unique(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


@dataclass(frozen=True)
class RetrievalCase:
    question_id: str
    relevant_chunk_ids: tuple[str, ...]
    ranked_chunk_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.question_id.strip():
            raise ValueError("question_id must not be empty")
        if not self.relevant_chunk_ids:
            raise ValueError("relevant_chunk_ids must contain at least one chunk")
        if any(not chunk_id.strip() for chunk_id in self.relevant_chunk_ids):
            raise ValueError("relevant_chunk_ids must not contain empty values")
        if any(not chunk_id.strip() for chunk_id in self.ranked_chunk_ids):
            raise ValueError("ranked_chunk_ids must not contain empty values")


@dataclass(frozen=True)
class RetrievalMetrics:
    case_count: int
    hit_at_1: float
    hit_at_3: float
    recall_at_k: float
    recall_k: int
    mrr: float

    def to_dict(self) -> dict[str, int | float]:
        return {
            "case_count": self.case_count,
            "hit_at_1": self.hit_at_1,
            "hit_at_3": self.hit_at_3,
            "recall_at_k": self.recall_at_k,
            "recall_k": self.recall_k,
            "mrr": self.mrr,
        }


def compute_retrieval_metrics(
    cases: tuple[RetrievalCase, ...],
    recall_k: int = 5,
) -> RetrievalMetrics:
    """Compute macro-averaged ranking metrics over a fixed question set."""

    if not cases:
        raise ValueError("at least one retrieval case is required")
    if recall_k <= 0:
        raise ValueError("recall_k must be greater than zero")

    hit_at_1 = 0
    hit_at_3 = 0
    recall_sum = 0.0
    reciprocal_rank_sum = 0.0
    for case in cases:
        relevant = set(case.relevant_chunk_ids)
        ranked = _unique(case.ranked_chunk_ids)
        hit_at_1 += bool(set(ranked[:1]) & relevant)
        hit_at_3 += bool(set(ranked[:3]) & relevant)
        recall_sum += len(set(ranked[:recall_k]) & relevant) / len(relevant)
        first_rank = next(
            (rank for rank, chunk_id in enumerate(ranked, start=1) if chunk_id in relevant),
            None,
        )
        if first_rank is not None:
            reciprocal_rank_sum += 1.0 / first_rank

    count = len(cases)
    return RetrievalMetrics(
        case_count=count,
        hit_at_1=hit_at_1 / count,
        hit_at_3=hit_at_3 / count,
        recall_at_k=recall_sum / count,
        recall_k=recall_k,
        mrr=reciprocal_rank_sum / count,
    )


def build_retrieval_report(
    *,
    strategy_cases: Mapping[str, tuple[RetrievalCase, ...]],
    git_commit: str,
    git_dirty: bool,
    corpus_hash: str,
    question_set_hash: str,
    retrieval_settings: Mapping[str, Any],
    runtime_environment: Mapping[str, str],
    recall_k: int,
) -> dict[str, Any]:
    """Build a JSON-serializable report with provenance and raw rankings."""

    if not strategy_cases:
        raise ValueError("at least one retrieval strategy is required")
    expected_judgments: tuple[tuple[str, tuple[str, ...]], ...] | None = None
    strategies: dict[str, Any] = {}
    for strategy_name, cases in strategy_cases.items():
        if not strategy_name.strip():
            raise ValueError("retrieval strategy names must not be empty")
        question_ids = [case.question_id for case in cases]
        if len(set(question_ids)) != len(question_ids):
            raise ValueError("a retrieval strategy contains a duplicate question id")
        judgments = tuple(
            (case.question_id, case.relevant_chunk_ids) for case in cases
        )
        if expected_judgments is None:
            expected_judgments = judgments
        elif judgments != expected_judgments:
            raise ValueError("every retrieval strategy must evaluate the same questions")
        metrics = compute_retrieval_metrics(cases, recall_k=recall_k)
        strategies[strategy_name] = {
            "metrics": metrics.to_dict(),
            "rankings": [
                {
                    "question_id": case.question_id,
                    "relevant_chunk_ids": list(case.relevant_chunk_ids),
                    "ranked_chunk_ids": list(_unique(case.ranked_chunk_ids)),
                }
                for case in cases
            ],
        }
    return {
        "schema_version": "1.0",
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "corpus_hash": corpus_hash,
        "question_set_hash": question_set_hash,
        "retrieval_settings": dict(retrieval_settings),
        "runtime_environment": dict(runtime_environment),
        "strategies": strategies,
    }
