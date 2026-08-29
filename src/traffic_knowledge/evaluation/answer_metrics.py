"""Deterministic citation and tool-routing metrics."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

TOOL_LABELS = {
    "search_traffic_knowledge": "knowledge",
    "get_model_metrics": "metrics",
    "run_traffic_forecast": "forecast",
}


def classify_tool_calls(tool_names: tuple[str, ...]) -> str:
    if not tool_names:
        return "none"
    labels = tuple(TOOL_LABELS.get(name, "unknown") for name in tool_names)
    if labels == ("knowledge", "metrics", "forecast"):
        return "combined"
    if len(labels) == 1:
        return labels[0]
    return "+".join(labels)


def build_benchmark_agent_input(question: str) -> dict[str, object]:
    """Supply bounded fallback forecast data so routing mistakes are recordable."""

    normalized = question.strip()
    if not normalized:
        raise ValueError("question must not be empty")
    return {
        "question": normalized,
        "forecast_model": "gru",
        "forecast_inputs": [[[[0.0]]]],
    }


@dataclass(frozen=True)
class AnswerEvaluationCase:
    question_id: str
    expected_tool: str
    selected_tool: str
    relevant_chunk_ids: tuple[str, ...]
    cited_chunk_ids: tuple[str, ...]
    tool_call_success: bool

    def __post_init__(self) -> None:
        if not all(
            value.strip()
            for value in (self.question_id, self.expected_tool, self.selected_tool)
        ):
            raise ValueError("question and tool identifiers must not be empty")
        if not self.relevant_chunk_ids:
            raise ValueError("relevant_chunk_ids must not be empty")


@dataclass(frozen=True)
class AnswerMetrics:
    case_count: int
    citation_correctness: float
    tool_selection_accuracy: float
    tool_call_success_rate: float
    tool_confusion_matrix: dict[str, dict[str, int]]

    def to_dict(self) -> dict[str, object]:
        return {
            "case_count": self.case_count,
            "citation_correctness": self.citation_correctness,
            "tool_selection_accuracy": self.tool_selection_accuracy,
            "tool_call_success_rate": self.tool_call_success_rate,
            "tool_confusion_matrix": self.tool_confusion_matrix,
        }


def compute_answer_metrics(cases: tuple[AnswerEvaluationCase, ...]) -> AnswerMetrics:
    if not cases:
        raise ValueError("at least one answer evaluation case is required")
    question_ids = [case.question_id for case in cases]
    if len(set(question_ids)) != len(question_ids):
        raise ValueError("answer evaluation contains a duplicate question id")

    citation_cases = [
        case for case in cases if case.expected_tool in {"knowledge", "combined"}
    ]
    citation_correctness = (
        sum(
            sum(
                chunk_id in set(case.relevant_chunk_ids)
                for chunk_id in case.cited_chunk_ids
            )
            / len(case.cited_chunk_ids)
            if case.cited_chunk_ids
            else 0.0
            for case in citation_cases
        )
        / len(citation_cases)
        if citation_cases
        else 0.0
    )
    confusion: defaultdict[str, defaultdict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    for case in cases:
        confusion[case.expected_tool][case.selected_tool] += 1
    count = len(cases)
    return AnswerMetrics(
        case_count=count,
        citation_correctness=citation_correctness,
        tool_selection_accuracy=sum(
            case.expected_tool == case.selected_tool for case in cases
        )
        / count,
        tool_call_success_rate=sum(case.tool_call_success for case in cases) / count,
        tool_confusion_matrix={
            expected: dict(selected) for expected, selected in confusion.items()
        },
    )
