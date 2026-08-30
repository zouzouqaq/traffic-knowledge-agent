import json
import subprocess
import sys
from pathlib import Path

from traffic_knowledge.evaluation.deepseek_runner import (
    build_deepseek_evaluation_report,
    summarize_deepseek_responses,
)


def _response(
    *,
    mode="deepseek",
    fallback=False,
    error_code=None,
    duration_ms=100.0,
    prompt_tokens=10,
    completion_tokens=5,
    citations=1,
):
    return {
        "answer": "sensitive generated answer",
        "citations": [{"chunk_id": str(index)} for index in range(citations)],
        "generation": {
            "answer_mode": mode,
            "answer_model": "deepseek-v4-flash" if mode == "deepseek" else None,
            "llm_fallback": fallback,
            "llm_error_code": error_code,
            "duration_ms": duration_ms,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        },
    }


def test_summarizes_success_fallback_latency_tokens_and_peak_cost():
    responses = (
        ("q-1", _response(duration_ms=100, citations=2)),
        (
            "q-2",
            _response(
                fallback=True,
                error_code="LLM_TIMEOUT",
                duration_ms=300,
                prompt_tokens=20,
                completion_tokens=10,
                citations=0,
            ),
        ),
    )

    summary, per_question = summarize_deepseek_responses(responses)

    assert summary["question_count"] == 2
    assert summary["deepseek_success_rate"] == 0.5
    assert summary["fallback_rate"] == 0.5
    assert summary["citation_presence_rate"] == 0.5
    assert summary["p50_duration_ms"] == 200.0
    assert summary["p95_duration_ms"] == 290.0
    assert summary["prompt_tokens"] == 30
    assert summary["completion_tokens"] == 15
    assert summary["estimated_peak_cost_usd"] == 0.000033
    assert per_question[1]["llm_error_code"] == "LLM_TIMEOUT"


def test_report_contains_provenance_but_never_answer_text_or_key():
    responses = (("q-1", _response()),)

    report = build_deepseek_evaluation_report(
        responses=responses,
        created_at_utc="2026-08-30T00:00:00Z",
        git_commit="abc123",
        git_dirty=False,
        git_working_tree_hash=None,
        questions_sha256="f" * 64,
        model="deepseek-v4-flash",
        base_url="https://api.deepseek.com/v1?token=secret",
    )
    serialized = json.dumps(report)

    assert report["configuration"]["base_url_host"] == "api.deepseek.com"
    assert report["per_question"][0]["question_id"] == "q-1"
    assert "answer" not in report["per_question"][0]
    assert "sensitive generated answer" not in serialized
    assert "secret" not in serialized
    assert "sk-" not in serialized


def test_deepseek_evaluation_cli_exposes_bounded_inputs():
    project_root = Path(__file__).resolve().parents[1]
    script = project_root / "scripts" / "evaluate_deepseek_answers.py"

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--api-url" in result.stdout
    assert "--questions-path" in result.stdout
    assert "--question-count" in result.stdout
    assert "--output-path" in result.stdout
