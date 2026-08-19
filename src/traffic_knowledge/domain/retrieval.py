"""Retrieval-domain result contracts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SearchHit:
    chunk_id: str
    document_id: str
    text: str
    location: str
    filename: str
    channels: tuple[str, ...]
    ranks: tuple[tuple[str, int], ...]
    score: float
