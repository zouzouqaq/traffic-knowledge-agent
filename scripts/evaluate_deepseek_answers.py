"""Evaluate live DeepSeek answer generation through the running API."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

import httpx

from traffic_knowledge.evaluation.dataset import (
    EvaluationQuestion,
    load_evaluation_questions,
)
from traffic_knowledge.evaluation.deepseek_runner import (
    build_deepseek_evaluation_report,
)
from traffic_knowledge.evaluation.provenance import file_sha256, git_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default="http://127.0.0.1:18100")
    parser.add_argument("--questions-path", type=Path, required=True)
    parser.add_argument("--question-count", type=int, default=10)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--timeout-seconds", type=float, default=45.0)
    return parser.parse_args()


def configured_deepseek_base_url(
    environment: Mapping[str, str] | None = None,
) -> str:
    source = os.environ if environment is None else environment
    base_url = source.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    normalized = base_url.strip().rstrip("/")
    if not normalized:
        raise ValueError("DEEPSEEK_BASE_URL must not be empty")
    return normalized


def select_knowledge_questions(
    questions: Sequence[EvaluationQuestion],
    *,
    count: int,
) -> tuple[EvaluationQuestion, ...]:
    selected = tuple(
        question for question in questions if question.expected_tool == "knowledge"
    )
    if count > len(selected):
        raise ValueError(
            f"question-count {count} exceeds knowledge question count {len(selected)}"
        )
    return selected[:count]


def main() -> None:
    args = parse_args()
    if args.question_count <= 0:
        raise ValueError("question-count must be positive")
    if args.timeout_seconds <= 0:
        raise ValueError("timeout-seconds must be positive")

    questions = select_knowledge_questions(
        load_evaluation_questions(args.questions_path),
        count=args.question_count,
    )

    responses = []
    with httpx.Client(
        base_url=args.api_url.rstrip("/"), timeout=args.timeout_seconds
    ) as client:
        for question in questions:
            response = client.post("/chat", json={"question": question.question})
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError("chat API response must be an object")
            responses.append((question.id, payload))

    project_root = Path(__file__).resolve().parents[1]
    commit, dirty, working_tree_hash = git_state(project_root)
    report = build_deepseek_evaluation_report(
        responses=tuple(responses),
        created_at_utc=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        git_commit=commit,
        git_dirty=dirty,
        git_working_tree_hash=working_tree_hash,
        questions_sha256=file_sha256(args.questions_path),
        model=args.model,
        base_url=configured_deepseek_base_url(),
    )
    serialized = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(serialized, encoding="utf-8")
    sys.stdout.write(serialized)
    print(f"Report saved to: {args.output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
