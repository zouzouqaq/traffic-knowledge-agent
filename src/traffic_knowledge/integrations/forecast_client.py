"""HTTP adapter for the external traffic forecasting service."""

from __future__ import annotations

from dataclasses import dataclass

import httpx
import numpy as np


class ForecastIntegrationError(RuntimeError):
    """Forecast-service error with a stable machine-readable code."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True)
class ForecastResult:
    model: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    predictions: list
    dataset: str | None = None
    inference_time_ms: float | None = None


class ForecastClient:
    def __init__(
        self,
        base_url: str,
        timeout_seconds: float,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not base_url.strip():
            raise ValueError("base_url must not be empty")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.transport = transport

    def forecast(
        self,
        model: str,
        inputs: list,
        target_time_features: list | None = None,
    ) -> ForecastResult:
        normalized_model = model.strip().lower()
        if not normalized_model or not normalized_model.replace("_", "").replace("-", "").isalnum():
            raise ValueError("model must contain only letters, numbers, or underscores")
        request_shape = tuple(np.asarray(inputs).shape)
        try:
            with httpx.Client(
                base_url=self.base_url,
                timeout=self.timeout_seconds,
                transport=self.transport,
            ) as client:
                request_payload = {"inputs": inputs}
                if target_time_features is not None:
                    request_payload["target_time_features"] = target_time_features
                response = client.post(
                    f"/v1/forecast/{normalized_model}",
                    json=request_payload,
                )
        except httpx.TimeoutException as error:
            raise ForecastIntegrationError("FORECAST_TIMEOUT", str(error)) from error
        except httpx.HTTPError as error:
            raise ForecastIntegrationError("FORECAST_UNAVAILABLE", str(error)) from error

        if response.status_code >= 400:
            raise ForecastIntegrationError(
                "FORECAST_UNAVAILABLE",
                f"forecast service returned HTTP {response.status_code}",
            )
        try:
            payload = response.json()
            result = _parse_result(payload)
        except (TypeError, ValueError, KeyError) as error:
            raise ForecastIntegrationError(
                "FORECAST_INVALID_RESPONSE", str(error)
            ) from error
        if result.model != normalized_model:
            raise ForecastIntegrationError(
                "FORECAST_INVALID_RESPONSE",
                "response model does not match requested model",
            )
        if result.input_shape != request_shape:
            raise ForecastIntegrationError(
                "FORECAST_INVALID_RESPONSE",
                "response input_shape does not match request inputs",
            )
        return result


def _parse_result(payload) -> ForecastResult:
    if not isinstance(payload, dict):
        raise TypeError("response must be a JSON object")
    model = payload["model"]
    input_shape = _parse_shape(payload["input_shape"], "input_shape")
    output_shape = _parse_shape(payload["output_shape"], "output_shape")
    predictions = payload["predictions"]
    if not isinstance(model, str) or not isinstance(predictions, list):
        raise TypeError("model must be a string and predictions must be a list")
    array = np.asarray(predictions, dtype=np.float64)
    if tuple(array.shape) != output_shape:
        raise ValueError("predictions shape does not match output_shape")
    if not np.isfinite(array).all():
        raise ValueError("predictions contain NaN or infinity")
    return ForecastResult(
        model=model,
        input_shape=input_shape,
        output_shape=output_shape,
        predictions=predictions,
        dataset=payload.get("dataset"),
        inference_time_ms=payload.get("inference_time_ms"),
    )


def _parse_shape(value, name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise TypeError(f"{name} must be a non-empty list")
    if any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in value):
        raise ValueError(f"{name} must contain positive integers")
    return tuple(value)
