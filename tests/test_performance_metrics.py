import threading
import time
from pathlib import Path

import pytest

from traffic_knowledge.evaluation.performance import (
    benchmark_callable,
    benchmark_throughput,
    directory_size_bytes,
    percentile,
)


def test_percentile_uses_linear_interpolation():
    values = (10.0, 20.0, 30.0, 40.0)

    assert percentile(values, 50) == pytest.approx(25.0)
    assert percentile(values, 95) == pytest.approx(38.5)


def test_benchmark_excludes_warmups_and_tracks_rss_delta():
    calls = []
    timestamps = iter((0, 20_000_000, 50_000_000, 80_000_000))
    rss_values = iter((100, 120, 110, 160))

    result = benchmark_callable(
        lambda: calls.append("call"),
        warmup_runs=1,
        measured_runs=2,
        clock_ns=lambda: next(timestamps),
        rss_bytes=lambda: next(rss_values),
        rss_sample_interval_seconds=None,
    )

    assert len(calls) == 3
    assert result.measured_runs == 2
    assert result.latencies_ms == (20.0, 30.0)
    assert result.p50_ms == pytest.approx(25.0)
    assert result.p95_ms == pytest.approx(29.5)
    assert result.sequential_requests_per_second == pytest.approx(40.0)
    assert result.baseline_rss_bytes == 100
    assert result.peak_rss_delta_bytes == 60


def test_directory_size_sums_nested_files(tmp_path: Path):
    (tmp_path / "a.bin").write_bytes(b"123")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "b.bin").write_bytes(b"12345")

    assert directory_size_bytes(tmp_path) == 8


def test_throughput_benchmark_uses_requested_concurrency():
    lock = threading.Lock()
    active = 0
    maximum_active = 0

    def operation():
        nonlocal active, maximum_active
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
        time.sleep(0.02)
        with lock:
            active -= 1

    result = benchmark_throughput(
        operation,
        request_count=6,
        concurrency=3,
    )

    assert maximum_active == 3
    assert result.request_count == 6
    assert result.concurrency == 3
    assert result.elapsed_seconds > 0
    assert result.requests_per_second > 0


def test_benchmark_samples_rss_while_operation_is_running():
    current_rss = 100

    def operation():
        nonlocal current_rss
        current_rss = 260
        time.sleep(0.03)
        current_rss = 120

    result = benchmark_callable(
        operation,
        warmup_runs=0,
        measured_runs=1,
        rss_bytes=lambda: current_rss,
        rss_sample_interval_seconds=0.001,
    )

    assert result.baseline_rss_bytes == 100
    assert result.peak_rss_delta_bytes == 160
