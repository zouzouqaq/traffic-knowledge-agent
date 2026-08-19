"""Citation mapping and strict answer-label validation."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from traffic_knowledge.domain.retrieval import SearchHit

_CITATION_TOKEN = re.compile(r"\[S[^\]]*\]")
_VALID_CITATION = re.compile(r"\[S([1-9][0-9]*)\]")
_STATEMENT_BOUNDARY = re.compile(
    r"(?:\r?\n)+|(?<=[.!?])\s+|(?<=[\u3002\uff01\uff1f])\s*"
)


class CitationValidationError(ValueError):
    """Raised when a generated answer cannot be resolved to retrieved evidence."""


@dataclass(frozen=True)
class Citation:
    label: str
    document_id: str
    filename: str
    location: str
    chunk_id: str
    excerpt: str


def _bounded_excerpt(text: str, max_characters: int) -> str:
    if max_characters < 4:
        raise ValueError("max_excerpt_characters must be at least 4")
    normalized = " ".join(text.split())
    if len(normalized) <= max_characters:
        return normalized
    return f"{normalized[: max_characters - 3]}..."


def build_citations(
    hits: Sequence[SearchHit], max_excerpt_characters: int = 240
) -> tuple[Citation, ...]:
    """Map ranked retrieval hits to stable source labels for one answer."""
    return tuple(
        Citation(
            label=f"S{index}",
            document_id=hit.document_id,
            filename=hit.filename,
            location=hit.location,
            chunk_id=hit.chunk_id,
            excerpt=_bounded_excerpt(hit.text, max_excerpt_characters),
        )
        for index, hit in enumerate(hits, start=1)
    )


def select_cited_sources(
    answer: str, citations: Sequence[Citation]
) -> tuple[Citation, ...]:
    """Return only referenced citations and reject unresolved labels."""
    tokens = _CITATION_TOKEN.findall(answer)
    malformed = [token for token in tokens if _VALID_CITATION.fullmatch(token) is None]
    if malformed:
        raise CitationValidationError(
            f"answer contains malformed citation labels: {', '.join(malformed)}"
        )
    labels = {
        f"S{match.group(1)}"
        for token in tokens
        if (match := _VALID_CITATION.fullmatch(token)) is not None
    }
    if not labels:
        raise CitationValidationError("answer must contain at least one source label")

    by_label = {citation.label: citation for citation in citations}
    unknown = sorted(labels - by_label.keys())
    if unknown:
        raise CitationValidationError(
            f"answer contains unknown citation labels: {', '.join(unknown)}"
        )
    statements = [part.strip() for part in _STATEMENT_BOUNDARY.split(answer) if part.strip()]
    if any(_VALID_CITATION.search(statement) is None for statement in statements):
        raise CitationValidationError("every statement must contain a source label")
    return tuple(citation for citation in citations if citation.label in labels)
