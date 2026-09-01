import json

import httpx

from traffic_knowledge.integrations.forecast_client import ForecastClient


def test_forecast_forwards_historical_average_time_features():
    inputs = [[[[1.0]], [[2.0]]]]
    target_times = [[[1, 8]]]

    def handler(request):
        assert request.url.path == "/v1/forecast/historical-average"
        assert json.loads(request.read()) == {
            "inputs": inputs,
            "target_time_features": target_times,
        }
        return httpx.Response(
            200,
            json={
                "model": "historical-average",
                "dataset": "pems04",
                "input_shape": [1, 2, 1, 1],
                "output_shape": [1, 1, 1, 1],
                "predictions": [[[[3.0]]]],
                "inference_time_ms": 0.2,
            },
        )

    client = ForecastClient(
        "http://127.0.0.1:18000",
        2,
        transport=httpx.MockTransport(handler),
    )
    result = client.forecast("historical-average", inputs, target_times)

    assert result.model == "historical-average"
    assert result.dataset == "pems04"
    assert result.inference_time_ms == 0.2
