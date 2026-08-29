"""Repeatable single-request latency and resource measurements."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path


def percentile(values: Sequence[float], percentage: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0 <= percentage <= 100:
        raise ValueError("percentage must be between 0 and 100")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * percentage / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


@dataclass(frozen=True)
class PerformanceMetrics:
    measured_runs: int
    latencies_ms: tuple[float, ...]
    p50_ms: float
    p95_ms: float
    sequential_requests_per_second: float
    baseline_rss_bytes: int
    peak_rss_delta_bytes: int

    def to_dict(self) -> dict[str, object]:
        return {
            "measured_runs": self.measured_runs,
            "latencies_ms": list(self.latencies_ms),
            "p50_ms": self.p50_ms,
            "p95_ms": self.p95_ms,
            "sequential_requests_per_second": self.sequential_requests_per_second,
            "baseline_rss_bytes": self.baseline_rss_bytes,
            "peak_rss_delta_bytes": self.peak_rss_delta_bytes,
        }


@dataclass(frozen=True)
class ThroughputMetrics:
    request_count: int
    concurrency: int
    elapsed_seconds: float
    requests_per_second: float

    def to_dict(self) -> dict[str, object]:
        return {
            "request_count": self.request_count,
            "concurrency": self.concurrency,
            "elapsed_seconds": self.elapsed_seconds,
            "requests_per_second": self.requests_per_second,
        }


def benchmark_callable(
    operation: Callable[[], object],
    *,
    warmup_runs: int,
    measured_runs: int,
    clock_ns: Callable[[], int] = time.perf_counter_ns,
    rss_bytes: Callable[[], int],
    rss_sample_interval_seconds: float | None = 0.01,
) -> PerformanceMetrics:
    if warmup_runs < 0 or measured_runs <= 0:
        raise ValueError("warmup_runs must be non-negative and measured_runs positive")
    if (
        rss_sample_interval_seconds is not None
        and rss_sample_interval_seconds <= 0
    ):
        raise ValueError("rss_sample_interval_seconds must be positive or None")
    baseline_rss = rss_bytes()
    rss_samples = [baseline_rss]
    stop_sampling = threading.Event()

    def sample_rss() -> None:
        assert rss_sample_interval_seconds is not None
        while not stop_sampling.wait(rss_sample_interval_seconds):
            rss_samples.append(rss_bytes())

    sampler = None
    if rss_sample_interval_seconds is not None:
        sampler = threading.Thread(target=sample_rss, daemon=True)
        sampler.start()
    try:
        for _ in range(warmup_runs):
            operation()
            rss_samples.append(rss_bytes())
        latencies = []
        for _ in range(measured_runs):
            started = clock_ns()
            operation()
            elapsed = clock_ns() - started
            latencies.append(elapsed / 1_000_000)
            rss_samples.append(rss_bytes())
    finally:
        stop_sampling.set()
        if sampler is not None:
            sampler.join()
    total_seconds = sum(latencies) / 1000
    return PerformanceMetrics(
        measured_runs=measured_runs,
        latencies_ms=tuple(latencies),
        p50_ms=percentile(latencies, 50),
        p95_ms=percentile(latencies, 95),
        sequential_requests_per_second=measured_runs / total_seconds,
        baseline_rss_bytes=baseline_rss,
        peak_rss_delta_bytes=max(rss_samples) - baseline_rss,
    )


def benchmark_throughput(
    operation: Callable[[], object],
    *,
    request_count: int,
    concurrency: int,
    clock_ns: Callable[[], int] = time.perf_counter_ns,
) -> ThroughputMetrics:
    if request_count <= 0:
        raise ValueError("request_count must be positive")
    if concurrency <= 0 or concurrency > request_count:
        raise ValueError("concurrency must be between one and request_count")

    started = clock_ns()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(operation) for _ in range(request_count)]
        for future in futures:
            future.result()
    elapsed_seconds = (clock_ns() - started) / 1_000_000_000
    if elapsed_seconds <= 0:
        raise ValueError("throughput benchmark elapsed time must be positive")
    return ThroughputMetrics(
        request_count=request_count,
        concurrency=concurrency,
        elapsed_seconds=elapsed_seconds,
        requests_per_second=request_count / elapsed_seconds,
    )


def directory_size_bytes(path: Path) -> int:
    root = Path(path)
    return sum(item.stat().st_size for item in root.rglob("*") if item.is_file())
