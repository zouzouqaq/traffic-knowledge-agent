"""Domain contracts for bounded traffic-agent execution."""

from __future__ import annotations

from dataclasses import dataclass

from traffic_knowledge.retrieval.citations import Citation


@dataclass(frozen=True)
class ToolCallRecord:
    name: str
    arguments: dict[str, object]
    duration_ms: float
    success: bool
    error_code: str | None = None


@dataclass(frozen=True)
class AgentError:
    code: str
    message: str
    tool: str


@dataclass(frozen=True)
class AgentResponse:
    answer: str
    citations: tuple[Citation, ...]
    tool_calls: tuple[ToolCallRecord, ...]
    partial: bool
    errors: tuple[AgentError, ...]
