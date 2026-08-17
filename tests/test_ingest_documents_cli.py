import json
import subprocess
import sys
from pathlib import Path


def test_cli_ingests_document_and_prints_json(tmp_path):
    document_path = tmp_path / "guide.md"
    document_path.write_text("# Traffic\nGRU predicts traffic flow.", encoding="utf-8")
    database_path = tmp_path / "metadata.sqlite3"
    project_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "ingest_documents.py"),
            str(document_path),
            "--database-path",
            str(database_path),
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout)
    assert output["status"] == "indexed"
    assert output["duplicate"] is False
    assert output["chunk_count"] == 1
    assert output["sha256"]
    assert database_path.is_file()
