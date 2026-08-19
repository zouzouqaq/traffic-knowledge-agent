"""Versioned, context-preserving traffic metrics snapshot reader."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


class MetricsSnapshotError(ValueError):
    """Metrics snapshot error with a stable machine-readable code."""

    def __init__(self, message: str) -> None:
        self.code = "METRICS_SCHEMA_INVALID"
        self.message = message
        super().__init__(f"{self.code}: {message}")


@dataclass(frozen=True)
class ForecastHorizon:
    steps: int
    interval_minutes: int


@dataclass(frozen=True)
class ModelMetrics:
    name: str
    mae: float
    rmse: float
    mape: float


@dataclass(frozen=True)
class MetricsSnapshot:
    schema_version: str
    dataset: str
    split: str
    horizon: ForecastHorizon
    created_at: str
    environment: dict[str, object]
    models: tuple[ModelMetrics, ...]


class MetricsSnapshotRepository:
    @staticmethod
    def load(path: Path) -> MetricsSnapshot:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return _parse_snapshot(payload)
        except MetricsSnapshotError:
            raise
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            raise MetricsSnapshotError(str(error)) from error


def _parse_snapshot(payload) -> MetricsSnapshot:
    if not isinstance(payload, dict):
        raise MetricsSnapshotError("snapshot must be a JSON object")
    if payload.get("schema_version") != "1.0":
        raise MetricsSnapshotError("unsupported schema_version")

    dataset = _nonempty_string(payload["dataset"], "dataset")
    split = _nonempty_string(payload["split"], "split")
    created_at = _nonempty_string(payload["created_at"], "created_at")
    parsed_created_at = datetime.fromisoformat(created_at)
    if "T" not in created_at or parsed_created_at.utcoffset() is None:
        raise MetricsSnapshotError(
            "created_at must be an ISO 8601 datetime with a UTC offset"
        )
    horizon = _parse_horizon(payload["horizon"])
    environment = payload["environment"]
    if not isinstance(environment, dict):
        raise MetricsSnapshotError("environment must be an object")
    raw_models = payload["models"]
    if not isinstance(raw_models, list) or not raw_models:
        raise MetricsSnapshotError("models must be a non-empty list")
    models = tuple(_parse_model(item) for item in raw_models)
    if len({model.name for model in models}) != len(models):
        raise MetricsSnapshotError("model names must be unique")
    return MetricsSnapshot(
        schema_version="1.0",
        dataset=dataset,
        split=split,
        horizon=horizon,
        created_at=created_at,
        environment=dict(environment),
        models=models,
    )


def _parse_horizon(value) -> ForecastHorizon:
    if not isinstance(value, dict):
        raise MetricsSnapshotError("horizon must be an object")
    steps = value["steps"]
    interval = value["interval_minutes"]
    if any(
        isinstance(item, bool) or not isinstance(item, int) or item <= 0
        for item in (steps, interval)
    ):
        raise MetricsSnapshotError("horizon values must be positive integers")
    return ForecastHorizon(steps=steps, interval_minutes=interval)


def _parse_model(value) -> ModelMetrics:
    if not isinstance(value, dict):
        raise MetricsSnapshotError("each model must be an object")
    name = _nonempty_string(value["name"], "model name")
    metrics = tuple(_finite_float(value[field], field) for field in ("mae", "rmse", "mape"))
    return ModelMetrics(name=name, mae=metrics[0], rmse=metrics[1], mape=metrics[2])


def _finite_float(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise MetricsSnapshotError(f"{name} must be numeric")
    converted = float(value)
    if not math.isfinite(converted):
        raise MetricsSnapshotError(f"{name} must be finite")
    if converted < 0:
        raise MetricsSnapshotError(f"{name} must not be negative")
    return converted


def _nonempty_string(value, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MetricsSnapshotError(f"{name} must be a non-empty string")
    return value.strip()
