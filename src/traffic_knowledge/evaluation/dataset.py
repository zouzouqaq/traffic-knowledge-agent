"""Validated JSONL question sets for traffic-knowledge evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

SUPPORTED_TOOLS = frozenset({"knowledge", "forecast", "metrics", "combined"})


class EvaluationDatasetError(ValueError):
    """Raised when an evaluation question file breaks its data contract."""


def _required_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvaluationDatasetError(f"{field_name} must be a non-empty string")
    return value.strip()


def _required_text_tuple(value: object, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise EvaluationDatasetError(f"{field_name} must contain at least one item")
    items = tuple(_required_text(item, field_name) for item in value)
    if len(set(items)) != len(items):
        raise EvaluationDatasetError(f"{field_name} must not contain duplicates")
    return items


@dataclass(frozen=True)
class EvaluationQuestion:
    id: str
    question: str
    category: str
    expected_answer_points: tuple[str, ...]
    relevant_chunk_ids: tuple[str, ...]
    expected_tool: str

    @classmethod
    def from_mapping(cls, value: object) -> EvaluationQuestion:
        if not isinstance(value, dict):
            raise EvaluationDatasetError("each JSONL row must be an object")
        expected_tool = _required_text(value.get("expected_tool"), "expected_tool")
        if expected_tool not in SUPPORTED_TOOLS:
            supported = ", ".join(sorted(SUPPORTED_TOOLS))
            raise EvaluationDatasetError(
                f"expected_tool must be one of: {supported}"
            )
        return cls(
            id=_required_text(value.get("id"), "id"),
            question=_required_text(value.get("question"), "question"),
            category=_required_text(value.get("category"), "category"),
            expected_answer_points=_required_text_tuple(
                value.get("expected_answer_points"), "expected_answer_points"
            ),
            relevant_chunk_ids=_required_text_tuple(
                value.get("relevant_chunk_ids"), "relevant_chunk_ids"
            ),
            expected_tool=expected_tool,
        )


def load_evaluation_questions(path: Path | str) -> tuple[EvaluationQuestion, ...]:
    """Read, validate and freeze a UTF-8 JSONL evaluation set."""

    source = Path(path)
    questions: list[EvaluationQuestion] = []
    seen_ids: set[str] = set()
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                question = EvaluationQuestion.from_mapping(payload)
            except (json.JSONDecodeError, EvaluationDatasetError) as error:
                raise EvaluationDatasetError(
                    f"invalid evaluation question at line {line_number}: {error}"
                ) from error
            if question.id in seen_ids:
                raise EvaluationDatasetError(
                    f"duplicate question id at line {line_number}: {question.id}"
                )
            seen_ids.add(question.id)
            questions.append(question)
    if not questions:
        raise EvaluationDatasetError("question file must contain at least one question")
    return tuple(questions)
