"""Domain contracts for bounded traffic-agent execution."""

from __future__ import annotations

from dataclasses import dataclass, field

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
class AnswerGenerationMetadata:
    answer_mode: str = "evidence"
    answer_model: str | None = None
    llm_fallback: bool = False
    llm_error_code: str | None = None
    duration_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass(frozen=True)
class AgentResponse:
    answer: str
    citations: tuple[Citation, ...]
    tool_calls: tuple[ToolCallRecord, ...]
    partial: bool
    errors: tuple[AgentError, ...]
    generation: AnswerGenerationMetadata = field(
        default_factory=AnswerGenerationMetadata
    )
