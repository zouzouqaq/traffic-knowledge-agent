"""Validated contracts for parsed source documents."""

from __future__ import annotations

from dataclasses import dataclass


class DocumentValidationError(ValueError):
    """A document error with a stable machine-readable code."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True)
class SourceBlock:
    """A section or page of source text with its original location."""

    text: str
    location: str
    ordinal: int


@dataclass(frozen=True)
class ParsedDocument:
    """One source file normalized into ordered text blocks."""

    filename: str
    media_type: str
    blocks: tuple[SourceBlock, ...]
