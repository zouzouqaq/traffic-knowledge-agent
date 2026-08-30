from pathlib import Path

import httpx
from streamlit.testing.v1 import AppTest

from traffic_knowledge.api.dependencies import build_answer_generator
from traffic_knowledge.application.grounded_answers import (
    EvidenceOnlyAnswerGenerator,
    ResilientDeepSeekAnswerGenerator,
)
from traffic_knowledge.settings import Settings


def _settings(answer_mode: str, api_key: str | None) -> Settings:
    return Settings(
        data_dir=Path("data"),
        database_path=Path("data/metadata.sqlite3"),
        chroma_path=Path("data/chroma"),
        forecast_base_url="http://127.0.0.1:18000",
        request_timeout_seconds=10,
        max_file_bytes=1024,
        retrieval_vector_weight=0.6,
        retrieval_bm25_weight=0.4,
        embedding_model_name="local-model",
        answer_mode=answer_mode,
        deepseek_api_key=api_key,
        deepseek_base_url="https://api.deepseek.com",
        deepseek_model="deepseek-v4-flash",
        deepseek_timeout_seconds=20,
        deepseek_temperature=0.2,
        deepseek_max_output_tokens=800,
    )


def test_answer_generator_defaults_to_evidence_only():
    generator = build_answer_generator(_settings("evidence", None))

    assert isinstance(generator, EvidenceOnlyAnswerGenerator)


def test_deepseek_generator_wraps_evidence_fallback():
    generator = build_answer_generator(
        _settings("deepseek", "test-key"),
        transport=httpx.MockTransport(lambda request: httpx.Response(503)),
    )

    assert isinstance(generator, ResilientDeepSeekAnswerGenerator)
    assert isinstance(generator.fallback, EvidenceOnlyAnswerGenerator)


def test_agent_result_displays_deepseek_generation_metadata():
    app = AppTest.from_string(
        '''
from streamlit_app import _render_agent_result
_render_agent_result({
    "answer": "回答 [S1]。", "citations": [], "partial": False, "errors": [],
    "generation": {"answer_mode": "deepseek", "answer_model": "deepseek-v4-flash",
                   "llm_fallback": False, "duration_ms": 321,
                   "prompt_tokens": 120, "completion_tokens": 18}
})
'''
    ).run(timeout=10)

    assert not app.exception
    captions = [item.value for item in app.caption]
    assert any("deepseek-v4-flash" in value for value in captions)
    assert any("tokens 120+18" in value for value in captions)


def test_agent_result_displays_safe_fallback_status():
    app = AppTest.from_string(
        '''
from streamlit_app import _render_agent_result
_render_agent_result({
    "answer": "模板回答 [S1]。", "citations": [], "partial": False, "errors": [],
    "generation": {"answer_mode": "evidence", "answer_model": None,
                   "llm_fallback": True, "llm_error_code": "LLM_TIMEOUT",
                   "duration_ms": 0, "prompt_tokens": 0, "completion_tokens": 0}
})
'''
    ).run(timeout=10)

    assert not app.exception
    assert any("已回退 (LLM_TIMEOUT)" in item.value for item in app.caption)
