import subprocess
import sys
from pathlib import Path


def test_benchmark_cli_exposes_fixed_run_configuration():
    project_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [sys.executable, str(project_root / "scripts" / "run_benchmark.py"), "--help"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--questions" in result.stdout
    assert "--corpus-dir" in result.stdout
    assert "--output" in result.stdout
    assert "--warmup-runs" in result.stdout
    assert "--measured-runs" in result.stdout
    assert "--throughput-concurrency" in result.stdout
    assert "--expected-question-count" in result.stdout
    assert "--metrics-path" in result.stdout
