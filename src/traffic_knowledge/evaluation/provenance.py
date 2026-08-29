"""Content fingerprints and runtime metadata for reproducible evaluations."""

from __future__ import annotations

import hashlib
import importlib.metadata
import platform
import subprocess
from collections.abc import Iterable, Sequence
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def directory_sha256(path: Path) -> str:
    root = Path(path)
    if not root.is_dir():
        raise ValueError("path must be a directory")
    digest = hashlib.sha256()
    for file_path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(file_path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_sha256(file_path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def corpus_sha256(chunks: Iterable[object]) -> str:
    digest = hashlib.sha256()
    for chunk in sorted(chunks, key=lambda item: item.chunk_id):
        fields = (
            chunk.chunk_id,
            chunk.document_id,
            str(chunk.ordinal),
            chunk.location,
            chunk.text,
        )
        digest.update("\x1f".join(fields).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def git_state(project_root: Path) -> tuple[str, bool, str | None]:
    root = Path(project_root)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        check=True,
    ).stdout.strip().decode("ascii")
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "-z"],
        cwd=root,
        capture_output=True,
        check=True,
    ).stdout
    if not status:
        return commit, False, None

    digest = hashlib.sha256(status)
    digest.update(
        subprocess.run(
            ["git", "diff", "HEAD", "--binary"],
            cwd=root,
            capture_output=True,
            check=True,
        ).stdout
    )
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        cwd=root,
        capture_output=True,
        check=True,
    ).stdout
    for relative in sorted(item for item in untracked.split(b"\0") if item):
        digest.update(relative)
        digest.update(b"\0")
        digest.update(file_sha256(root / relative.decode()).encode("ascii"))
    return commit, True, digest.hexdigest()


def runtime_environment(packages: Sequence[str]) -> dict[str, str]:
    versions = {name: importlib.metadata.version(name) for name in packages}
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": platform.processor() or "unknown",
        "device": "cpu",
        **{f"package:{name}": version for name, version in versions.items()},
    }
