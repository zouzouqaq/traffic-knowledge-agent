import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from traffic_knowledge.evaluation.runner import validate_retrieval_configuration
from traffic_knowledge.ingestion.chunking import DocumentChunk
from traffic_knowledge.ingestion.repository import DocumentRepository


def test_evaluation_cli_exposes_reproducible_inputs():
    project_root = Path(__file__).resolve().parents[1]
    script_path = project_root / "scripts" / "evaluate_retrieval.py"

    result = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--questions" in result.stdout
    assert "--output" in result.stdout
    assert "--data-dir" in result.stdout
    assert "--embedding-model" in result.stdout
    assert "--top-k" in result.stdout


@pytest.mark.parametrize(
    ("vector_weight", "bm25_weight"),
    [(float("nan"), 0.4), (0.6, float("inf")), (-0.1, 0.4), (0.0, 0.0)],
)
def test_rejects_invalid_retrieval_weights(vector_weight, bm25_weight):
    with pytest.raises(ValueError, match="weights"):
        validate_retrieval_configuration(
            top_k=5,
            vector_weight=vector_weight,
            bm25_weight=bm25_weight,
            rrf_constant=60,
        )


def test_cli_runs_three_strategies_and_writes_parseable_report(tmp_path):
    model_path = Path(
        os.getenv(
            "TRAFFIC_TEST_EMBEDDING_MODEL",
            "/8t/usr/zhouh2024/models/bge-small-zh-v1.5",
        )
    )
    if not model_path.is_dir():
        pytest.skip("local integration embedding model is unavailable")

    data_dir = tmp_path / "data"
    repository = DocumentRepository(data_dir / "metadata.sqlite3")
    repository.initialize()
    document = repository.begin_ingestion("a" * 64, "traffic.md")
    chunks = (
        DocumentChunk(
            chunk_id="chunk-gru",
            document_id=document.document_id,
            text="GRU uses recurrent gates for traffic flow prediction.",
            location="Models > GRU",
            ordinal=0,
            token_estimate=10,
        ),
        DocumentChunk(
            chunk_id="chunk-stdn",
            document_id=document.document_id,
            text="STDN separates spatial and temporal traffic dependencies.",
            location="Models > STDN",
            ordinal=1,
            token_estimate=10,
        ),
    )
    repository.replace_chunks(document.document_id, chunks)
    repository.mark_indexed(document.document_id)
    questions_path = tmp_path / "questions.jsonl"
    questions_path.write_text(
        json.dumps(
            {
                "id": "q-gru",
                "question": "Which model uses recurrent gates?",
                "category": "model_mechanisms",
                "expected_answer_points": ["GRU uses recurrent gates"],
                "relevant_chunk_ids": ["chunk-gru"],
                "expected_tool": "knowledge",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "retrieval_metrics.json"
    project_root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    conda_lib = Path(sys.base_prefix) / "lib"
    environment["LD_LIBRARY_PATH"] = os.pathsep.join(
        value
        for value in (str(conda_lib), environment.get("LD_LIBRARY_PATH", ""))
        if value
    )

    result = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "evaluate_retrieval.py"),
            "--questions",
            str(questions_path),
            "--output",
            str(output_path),
            "--data-dir",
            str(data_dir),
            "--embedding-model",
            str(model_path),
            "--top-k",
            "3",
        ],
        cwd=project_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    stdout_report = json.loads(result.stdout)
    file_report = json.loads(output_path.read_text(encoding="utf-8"))
    assert stdout_report == file_report
    assert set(file_report["strategies"]) == {"vector", "bm25", "hybrid"}
    assert file_report["strategies"]["vector"]["metrics"]["case_count"] == 1
    assert len(file_report["retrieval_settings"]["embedding_model_hash"]) == 64
    assert isinstance(file_report["git_dirty"], bool)
    assert "Report saved to:" in result.stderr
