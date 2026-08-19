import json

import httpx
import pytest

from traffic_knowledge.integrations.forecast_client import (
    ForecastClient,
    ForecastIntegrationError,
)


def _client(handler):
    return ForecastClient(
        base_url="http://127.0.0.1:18000",
        timeout_seconds=2,
        transport=httpx.MockTransport(handler),
    )


def test_forecast_sends_external_contract_and_returns_typed_result():
    inputs = [[[[1.0]], [[2.0]]]]

    def handler(request):
        assert request.method == "POST"
        assert request.url.path == "/v1/forecast/gru"
        assert json.loads(request.read()) == {"inputs": inputs}
        return httpx.Response(
            200,
            json={
                "model": "gru",
                "input_shape": [1, 2, 1, 1],
                "output_shape": [1, 1, 1, 1],
                "predictions": [[[[3.0]]]],
            },
        )

    result = _client(handler).forecast("gru", inputs)

    assert result.model == "gru"
    assert result.input_shape == (1, 2, 1, 1)
    assert result.output_shape == (1, 1, 1, 1)
    assert result.predictions == [[[[3.0]]]]


def test_forecast_maps_timeout_to_stable_error_code():
    def handler(request):
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(ForecastIntegrationError) as captured:
        _client(handler).forecast("gru", [[[[1.0]]]])

    assert captured.value.code == "FORECAST_TIMEOUT"


def test_forecast_maps_http_503_to_unavailable():
    def handler(request):
        return httpx.Response(503, json={"detail": "not ready"})

    with pytest.raises(ForecastIntegrationError) as captured:
        _client(handler).forecast("gru", [[[[1.0]]]])

    assert captured.value.code == "FORECAST_UNAVAILABLE"


@pytest.mark.parametrize(
    "payload",
    [
        {"model": "gru", "input_shape": [1], "output_shape": [1]},
        {
            "model": "gru",
            "input_shape": [1],
            "output_shape": [1],
            "predictions": [float("nan")],
        },
        {
            "model": "gru",
            "input_shape": "wrong",
            "output_shape": [1],
            "predictions": [1.0],
        },
    ],
)
def test_forecast_rejects_malformed_or_nonfinite_response(payload):
    def handler(request):
        return httpx.Response(200, json=payload)

    with pytest.raises(ForecastIntegrationError) as captured:
        _client(handler).forecast("gru", [[[[1.0]]]])

    assert captured.value.code == "FORECAST_INVALID_RESPONSE"


def test_forecast_rejects_response_input_shape_that_does_not_match_request():
    def handler(request):
        return httpx.Response(
            200,
            json={
                "model": "gru",
                "input_shape": [1, 99, 1, 1],
                "output_shape": [1, 1, 1, 1],
                "predictions": [[[[3.0]]]],
            },
        )

    with pytest.raises(ForecastIntegrationError) as captured:
        _client(handler).forecast("gru", [[[[1.0]], [[2.0]]]])

    assert captured.value.code == "FORECAST_INVALID_RESPONSE"
