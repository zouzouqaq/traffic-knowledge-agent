"""Privacy-preserving metrics for live DeepSeek answer evaluations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from urllib.parse import urlsplit

from traffic_knowledge.evaluation.performance import percentile

INPUT_PRICE_USD_PER_MILLION = 0.44
OUTPUT_PRICE_USD_PER_MILLION = 1.32
PRICE_SOURCE_URL = "https://api-docs.deepseek.com/quick_start/pricing"


def summarize_deepseek_responses(
    responses: Sequence[tuple[str, Mapping[str, object]]],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    if not responses:
        raise ValueError("responses must contain at least one item")

    per_question: list[dict[str, object]] = []
    durations: list[float] = []
    success_count = 0
    fallback_count = 0
    citation_count = 0
    prompt_tokens = 0
    completion_tokens = 0

    for question_id, response in responses:
        generation = response.get("generation")
        if not isinstance(generation, Mapping):
            raise ValueError("response generation metadata is missing")
        citations = response.get("citations", [])
        if not isinstance(citations, list):
            raise ValueError("response citations must be a list")

        mode = str(generation.get("answer_mode", "evidence"))
        fallback = bool(generation.get("llm_fallback", False))
        duration = float(generation.get("duration_ms", 0.0))
        prompt = int(generation.get("prompt_tokens", 0))
        completion = int(generation.get("completion_tokens", 0))
        success = mode == "deepseek" and not fallback

        success_count += int(success)
        fallback_count += int(fallback)
        citation_count += int(bool(citations))
        durations.append(duration)
        prompt_tokens += prompt
        completion_tokens += completion
        per_question.append(
            {
                "question_id": question_id,
                "deepseek_success": success,
                "fallback": fallback,
                "llm_error_code": generation.get("llm_error_code"),
                "citation_count": len(citations),
                "duration_ms": duration,
                "prompt_tokens": prompt,
                "completion_tokens": completion,
            }
        )

    count = len(responses)
    estimated_cost = (
        prompt_tokens * INPUT_PRICE_USD_PER_MILLION
        + completion_tokens * OUTPUT_PRICE_USD_PER_MILLION
    ) / 1_000_000
    summary = {
        "question_count": count,
        "deepseek_success_count": success_count,
        "deepseek_success_rate": success_count / count,
        "fallback_count": fallback_count,
        "fallback_rate": fallback_count / count,
        "citation_presence_count": citation_count,
        "citation_presence_rate": citation_count / count,
        "p50_duration_ms": percentile(durations, 50),
        "p95_duration_ms": percentile(durations, 95),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "estimated_peak_cost_usd": round(estimated_cost, 9),
    }
    return summary, per_question


def build_deepseek_evaluation_report(
    *,
    responses: Sequence[tuple[str, Mapping[str, object]]],
    created_at_utc: str,
    git_commit: str,
    git_dirty: bool,
    git_working_tree_hash: str | None,
    questions_sha256: str,
    model: str,
    base_url: str,
) -> dict[str, object]:
    summary, per_question = summarize_deepseek_responses(responses)
    return {
        "schema_version": "1.0",
        "created_at_utc": created_at_utc,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "git_working_tree_hash": git_working_tree_hash,
        "input_fingerprints": {"questions_sha256": questions_sha256},
        "configuration": {
            "model": model,
            "base_url_host": urlsplit(base_url).hostname,
        },
        "pricing_snapshot": {
            "input_usd_per_million_tokens": INPUT_PRICE_USD_PER_MILLION,
            "output_usd_per_million_tokens": OUTPUT_PRICE_USD_PER_MILLION,
            "cost_semantics": "peak estimate using uncached input price",
            "source_url": PRICE_SOURCE_URL,
        },
        "summary": summary,
        "per_question": per_question,
    }
