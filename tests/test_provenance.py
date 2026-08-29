import subprocess
from pathlib import Path
from types import SimpleNamespace

from traffic_knowledge.evaluation.provenance import (
    corpus_sha256,
    directory_sha256,
    file_sha256,
    git_state,
    runtime_environment,
)


def test_file_hash_changes_with_content(tmp_path: Path):
    path = tmp_path / "questions.jsonl"
    path.write_text("first", encoding="utf-8")
    first_hash = file_sha256(path)

    path.write_text("second", encoding="utf-8")

    assert file_sha256(path) != first_hash


def test_directory_hash_includes_relative_paths_and_contents(tmp_path: Path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    first_hash = directory_sha256(model_dir)

    nested = model_dir / "weights"
    nested.mkdir()
    (nested / "model.bin").write_bytes(b"weights")

    assert directory_sha256(model_dir) != first_hash


def test_corpus_hash_is_independent_of_input_order():
    first = SimpleNamespace(
        chunk_id="b", document_id="doc", ordinal=1, location="L2", text="two"
    )
    second = SimpleNamespace(
        chunk_id="a", document_id="doc", ordinal=0, location="L1", text="one"
    )

    assert corpus_sha256((first, second)) == corpus_sha256((second, first))


def test_git_state_fingerprints_tracked_and_untracked_changes(tmp_path: Path):
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"], cwd=tmp_path, check=True
    )
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("clean", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=tmp_path, check=True)

    clean = git_state(tmp_path)
    tracked.write_text("dirty", encoding="utf-8")
    (tmp_path / "untracked.txt").write_text("new", encoding="utf-8")
    dirty = git_state(tmp_path)

    assert clean[1:] == (False, None)
    assert dirty[0] == clean[0]
    assert dirty[1] is True
    assert dirty[2] is not None


def test_runtime_environment_records_requested_package_version():
    environment = runtime_environment(("numpy",))

    assert environment["device"] == "cpu"
    assert environment["package:numpy"]
