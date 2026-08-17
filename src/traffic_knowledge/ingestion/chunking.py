"""Deterministic, section-aware document chunking."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from traffic_knowledge.domain.document import ParsedDocument

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?\u3002\uFF01\uFF1F])\s+")


@dataclass(frozen=True)
class DocumentChunk:
    """One stable retrieval unit derived from a parsed source block."""

    chunk_id: str
    document_id: str
    text: str
    location: str
    ordinal: int
    token_estimate: int


def _sentences(text: str) -> list[str]:
    normalized = " ".join(text.split())
    if not normalized:
        return []
    return [part.strip() for part in _SENTENCE_BOUNDARY.split(normalized) if part.strip()]


def _block_chunks(text: str, max_characters: int, overlap_characters: int) -> list[str]:
    chunks: list[str] = []
    current = ""

    for sentence in _sentences(text):
        candidate = f"{current} {sentence}".strip() if current else sentence
        if not current or len(candidate) <= max_characters:
            current = candidate
            continue

        chunks.append(current)
        if len(sentence) > max_characters:
            current = sentence
            continue

        overlap = current[-overlap_characters:] if overlap_characters else ""
        candidate = f"{overlap} {sentence}".strip()
        current = candidate if len(candidate) <= max_characters else sentence

    if current:
        chunks.append(current)
    return chunks


def chunk_document(
    document_id: str,
    parsed: ParsedDocument,
    max_characters: int,
    overlap_characters: int,
) -> tuple[DocumentChunk, ...]:
    """Split parsed blocks into bounded chunks with reproducible identifiers."""
    if not document_id.strip():
        raise ValueError("document_id must not be empty")
    if max_characters <= 0:
        raise ValueError("max_characters must be greater than zero")
    if overlap_characters < 0 or overlap_characters >= max_characters:
        raise ValueError("overlap_characters must be between zero and max_characters")

    chunks: list[DocumentChunk] = []
    for block in parsed.blocks:
        for block_chunk_ordinal, text in enumerate(
            _block_chunks(block.text, max_characters, overlap_characters)
        ):
            payload = (
                f"{document_id}:{block.ordinal}:{block_chunk_ordinal}:{text}".encode()
            )
            chunks.append(
                DocumentChunk(
                    chunk_id=hashlib.sha256(payload).hexdigest(),
                    document_id=document_id,
                    text=text,
                    location=block.location,
                    ordinal=len(chunks),
                    token_estimate=max(1, (len(text) + 3) // 4),
                )
            )
    return tuple(chunks)
