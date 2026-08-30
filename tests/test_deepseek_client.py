import json

import httpx
import pytest

from traffic_knowledge.integrations.deepseek import (
    DeepSeekClient,
    DeepSeekClientError,
)


def _valid_response(content="基于证据的答案 [S1]。"):
    return {
        "model": "deepseek-v4-flash",
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": 120, "completion_tokens": 18},
    }


def _client(transport):
    return DeepSeekClient(
        api_key="test-key",
        base_url="https://api.deepseek.com",
        model="deepseek-v4-flash",
        timeout_seconds=20,
        temperature=0.2,
        max_output_tokens=800,
        transport=transport,
    )


def test_client_disables_thinking_and_parses_usage():
    captured = []

    def handler(request):
        captured.append(request)
        return httpx.Response(200, json=_valid_response())

    result = _client(httpx.MockTransport(handler)).generate("system", "evidence")
    payload = json.loads(captured[0].read())

    assert captured[0].url.path == "/chat/completions"
    assert captured[0].headers["Authorization"] == "Bearer test-key"
    assert payload["thinking"] == {"type": "disabled"}
    assert payload["temperature"] == pytest.approx(0.2)
    assert payload["max_tokens"] == 800
    assert "test-key" not in json.dumps(payload)
    assert result.content == "基于证据的答案 [S1]。"
    assert result.model == "deepseek-v4-flash"
    assert result.prompt_tokens == 120
    assert result.completion_tokens == 18
    assert result.duration_ms >= 0


@pytest.mark.parametrize(
    ("status", "code"),
    [(401, "LLM_AUTH_FAILED"), (429, "LLM_RATE_LIMITED"), (503, "LLM_UNAVAILABLE")],
)
def test_client_maps_http_errors(status, code):
    client = _client(
        httpx.MockTransport(lambda request: httpx.Response(status, request=request))
    )

    with pytest.raises(DeepSeekClientError) as captured:
        client.generate("system", "evidence")

    assert captured.value.code == code
    assert "test-key" not in str(captured.value)


def test_client_retries_one_server_error_then_succeeds():
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(503, request=request)
        return httpx.Response(200, json=_valid_response(), request=request)

    result = _client(httpx.MockTransport(handler)).generate("system", "evidence")

    assert calls == 2
    assert result.content


def test_client_maps_repeated_timeout_without_leaking_key():
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("upstream timeout", request=request)

    with pytest.raises(DeepSeekClientError) as captured:
        _client(httpx.MockTransport(handler)).generate("system", "evidence")

    assert calls == 2
    assert captured.value.code == "LLM_TIMEOUT"
    assert "test-key" not in str(captured.value)


def test_client_rejects_empty_answer():
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            json=_valid_response(content="   "),
            request=request,
        )
    )

    with pytest.raises(DeepSeekClientError) as captured:
        _client(transport).generate("system", "evidence")

    assert captured.value.code == "LLM_INVALID_RESPONSE"
