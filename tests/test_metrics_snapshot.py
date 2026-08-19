import json
from pathlib import Path

import pytest

from traffic_knowledge.integrations.metrics_snapshot import (
    MetricsSnapshotError,
    MetricsSnapshotRepository,
)

FIXTURE = Path(__file__).parent / "fixtures" / "metrics_snapshot.json"


def test_loads_versioned_metrics_with_comparison_context():
    snapshot = MetricsSnapshotRepository.load(FIXTURE)

    assert snapshot.schema_version == "1.0"
    assert snapshot.dataset == "PEMS04"
    assert snapshot.split == "test"
    assert snapshot.horizon.steps == 12
    assert snapshot.horizon.interval_minutes == 5
    assert snapshot.created_at == "2026-08-19T13:30:00+08:00"
    assert snapshot.environment["device"] == "NVIDIA GeForce RTX 4090"
    assert [model.name for model in snapshot.models] == [
        "gru",
        "historical_average",
    ]
    assert snapshot.models[0].mae == pytest.approx(27.136278)


def _write_snapshot(tmp_path, mutate):
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    mutate(payload)
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_rejects_unsupported_schema_version(tmp_path):
    path = _write_snapshot(tmp_path, lambda payload: payload.update(schema_version="2.0"))

    with pytest.raises(MetricsSnapshotError) as captured:
        MetricsSnapshotRepository.load(path)

    assert captured.value.code == "METRICS_SCHEMA_INVALID"


@pytest.mark.parametrize("field", ["dataset", "split", "horizon"])
def test_rejects_missing_comparison_context(tmp_path, field):
    path = _write_snapshot(tmp_path, lambda payload: payload.pop(field))

    with pytest.raises(MetricsSnapshotError) as captured:
        MetricsSnapshotRepository.load(path)

    assert captured.value.code == "METRICS_SCHEMA_INVALID"


@pytest.mark.parametrize("metric", [float("nan"), float("inf"), -float("inf")])
def test_rejects_nonfinite_metrics(tmp_path, metric):
    def mutate(payload):
        payload["models"][0]["mae"] = metric

    path = _write_snapshot(tmp_path, mutate)

    with pytest.raises(MetricsSnapshotError) as captured:
        MetricsSnapshotRepository.load(path)

    assert captured.value.code == "METRICS_SCHEMA_INVALID"


@pytest.mark.parametrize("metric_name", ["mae", "rmse", "mape"])
def test_rejects_negative_error_metrics(tmp_path, metric_name):
    def mutate(payload):
        payload["models"][0][metric_name] = -0.1

    path = _write_snapshot(tmp_path, mutate)

    with pytest.raises(MetricsSnapshotError) as captured:
        MetricsSnapshotRepository.load(path)

    assert captured.value.code == "METRICS_SCHEMA_INVALID"


@pytest.mark.parametrize(
    "created_at",
    ["2026-08-19", "2026-08-19T13:30:00"],
)
def test_rejects_created_at_without_time_or_utc_offset(tmp_path, created_at):
    path = _write_snapshot(
        tmp_path,
        lambda payload: payload.update(created_at=created_at),
    )

    with pytest.raises(MetricsSnapshotError) as captured:
        MetricsSnapshotRepository.load(path)

    assert captured.value.code == "METRICS_SCHEMA_INVALID"
