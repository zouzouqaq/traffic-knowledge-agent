"""Bounded LangGraph orchestration for traffic-domain tools."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, TypedDict

import numpy as np
from langgraph.graph import END, START, StateGraph

from traffic_knowledge.application.question_answering import AnswerResult
from traffic_knowledge.domain.agent import AgentError, AgentResponse, ToolCallRecord
from traffic_knowledge.integrations.forecast_client import (
    ForecastIntegrationError,
    ForecastResult,
)
from traffic_knowledge.integrations.metrics_snapshot import (
    MetricsSnapshot,
    MetricsSnapshotError,
)
from traffic_knowledge.retrieval.citations import CitationValidationError

AgentIntent = Literal["knowledge", "metrics", "forecast", "combined"]
TOOL_NAMES = (
    "search_traffic_knowledge",
    "get_model_metrics",
    "run_traffic_forecast",
)


class IntentModel(Protocol):
    def classify(self, question: str) -> str: ...


@dataclass(frozen=True)
class AgentDependencies:
    intent_model: IntentModel
    qa_service: object
    forecast_client: object
    metrics_repository: object
    metrics_path: Path
    max_tool_calls: int = 3

    def __post_init__(self) -> None:
        if self.max_tool_calls <= 0:
            raise ValueError("max_tool_calls must be greater than zero")


class AgentState(TypedDict, total=False):
    question: str
    forecast_model: str
    forecast_inputs: list
    intent: AgentIntent
    knowledge_result: AnswerResult
    metrics_result: MetricsSnapshot
    forecast_result: ForecastResult
    tool_calls: list[ToolCallRecord]
    errors: list[AgentError]
    response: AgentResponse


def search_traffic_knowledge(qa_service, question: str) -> AnswerResult:
    """Answer one traffic-domain question from indexed evidence."""
    normalized = question.strip()
    if not normalized:
        raise ValueError("question must not be empty")
    return qa_service.answer(normalized)


def get_model_metrics(metrics_repository, path: Path) -> MetricsSnapshot:
    """Load one versioned model-comparison snapshot."""
    return metrics_repository.load(path)


def run_traffic_forecast(forecast_client, model: str, inputs: list) -> ForecastResult:
    """Run one forecast using the external forecasting service contract."""
    normalized_model = model.strip().lower()
    if not normalized_model:
        raise ValueError("forecast_model must not be empty")
    if not isinstance(inputs, list) or not inputs:
        raise ValueError("forecast_inputs must be a non-empty list")
    if np.asarray(inputs).ndim != 4:
        raise ValueError("forecast_inputs must be a four-dimensional array")
    return forecast_client.forecast(normalized_model, inputs)


def build_agent_graph(dependencies: AgentDependencies):
    """Compile a fixed, non-looping graph with at most three tool calls."""
    builder = StateGraph(AgentState)

    def classify_intent(state: AgentState) -> dict:
        question = state.get("question", "").strip()
        if not question:
            raise ValueError("question must not be empty")
        intent = dependencies.intent_model.classify(question).strip().lower()
        if intent not in {"knowledge", "metrics", "forecast", "combined"}:
            raise ValueError(f"unsupported agent intent: {intent}")
        return {"question": question, "intent": intent, "tool_calls": [], "errors": []}

    def knowledge_node(state: AgentState) -> dict:
        return _execute_tools(state, dependencies, ("knowledge",))

    def metrics_node(state: AgentState) -> dict:
        return _execute_tools(state, dependencies, ("metrics",))

    def forecast_node(state: AgentState) -> dict:
        return _execute_tools(state, dependencies, ("forecast",))

    def combined_node(state: AgentState) -> dict:
        return _execute_tools(
            state,
            dependencies,
            ("knowledge", "metrics", "forecast"),
        )

    def compose_grounded_answer(state: AgentState) -> dict:
        parts = []
        citations = ()
        knowledge = state.get("knowledge_result")
        if knowledge is not None:
            parts.append(knowledge.answer)
            citations = knowledge.citations
        metrics = state.get("metrics_result")
        if metrics is not None:
            values = "; ".join(
                f"{item.name}: MAE={item.mae:.3f}, RMSE={item.rmse:.3f}, "
                f"MAPE={item.mape:.2f}%"
                for item in metrics.models
            )
            parts.append(
                f"{metrics.dataset} {metrics.split} 集的 {metrics.horizon.steps} 步"
                f"模型指标为: {values}。"
            )
        forecast = state.get("forecast_result")
        if forecast is not None:
            prediction_values = np.asarray(forecast.predictions, dtype=np.float64)
            parts.append(
                f"{forecast.model} 预测已完成, 输出形状为 {forecast.output_shape}, "
                f"预测值范围 {prediction_values.min():.3f}--"
                f"{prediction_values.max():.3f}, 均值 {prediction_values.mean():.3f}。"
            )
        if not parts:
            parts.append("当前请求未能获得可用结果。")
        errors = tuple(state.get("errors", []))
        return {
            "response": AgentResponse(
                answer="\n".join(parts),
                citations=tuple(citations),
                tool_calls=tuple(state.get("tool_calls", [])),
                partial=bool(errors),
                errors=errors,
            )
        }

    builder.add_node("classify_intent", classify_intent)
    builder.add_node("knowledge", knowledge_node)
    builder.add_node("metrics", metrics_node)
    builder.add_node("forecast", forecast_node)
    builder.add_node("combined", combined_node)
    builder.add_node("compose_grounded_answer", compose_grounded_answer)
    builder.add_edge(START, "classify_intent")
    builder.add_conditional_edges(
        "classify_intent",
        lambda state: state["intent"],
        {
            "knowledge": "knowledge",
            "metrics": "metrics",
            "forecast": "forecast",
            "combined": "combined",
        },
    )
    for node in ("knowledge", "metrics", "forecast", "combined"):
        builder.add_edge(node, "compose_grounded_answer")
    builder.add_edge("compose_grounded_answer", END)
    return builder.compile()


def _execute_tools(
    state: AgentState,
    dependencies: AgentDependencies,
    requested_tools: tuple[str, ...],
) -> dict:
    updates: dict[str, object] = {}
    calls = list(state.get("tool_calls", []))
    errors = list(state.get("errors", []))
    for tool in requested_tools:
        if len(calls) >= dependencies.max_tool_calls:
            errors.append(
                AgentError(
                    code="AGENT_TOOL_LIMIT",
                    message="maximum tool-call count reached",
                    tool=tool,
                )
            )
            break
        result_key, result, record, error = _call_tool(tool, state, dependencies)
        calls.append(record)
        if result is not None:
            updates[result_key] = result
        if error is not None:
            errors.append(error)
    updates["tool_calls"] = calls
    updates["errors"] = errors
    return updates


def _call_tool(tool: str, state: AgentState, dependencies: AgentDependencies):
    started = time.perf_counter_ns()
    try:
        if tool == "knowledge":
            name = "search_traffic_knowledge"
            arguments = {"question": state["question"][:200]}
            result_key = "knowledge_result"
            result = search_traffic_knowledge(dependencies.qa_service, state["question"])
        elif tool == "metrics":
            name = "get_model_metrics"
            arguments = {"path": dependencies.metrics_path.name}
            result_key = "metrics_result"
            result = get_model_metrics(
                dependencies.metrics_repository,
                dependencies.metrics_path,
            )
        else:
            name = "run_traffic_forecast"
            if "forecast_inputs" not in state:
                raise ValueError("forecast_inputs are required for forecast requests")
            model = state.get("forecast_model", "gru")
            inputs = state["forecast_inputs"]
            arguments = {
                "model": model,
                "input_shape": tuple(np.asarray(inputs).shape),
            }
            result_key = "forecast_result"
            result = run_traffic_forecast(dependencies.forecast_client, model, inputs)
        return (
            result_key,
            result,
            ToolCallRecord(
                name=name,
                arguments=arguments,
                duration_ms=_elapsed_ms(started),
                success=True,
            ),
            None,
        )
    except (
        ForecastIntegrationError,
        MetricsSnapshotError,
        CitationValidationError,
    ) as exception:
        return _failed_tool_result(tool, exception, locals().get("arguments", {}), started)
    except ValueError:
        raise
    except Exception as exception:
        return _failed_tool_result(tool, exception, locals().get("arguments", {}), started)


def _failed_tool_result(
    tool: str,
    exception: Exception,
    arguments: dict[str, object],
    started_ns: int,
):
    name = {
        "knowledge": "search_traffic_knowledge",
        "metrics": "get_model_metrics",
        "forecast": "run_traffic_forecast",
    }[tool]
    code, message = _error_details(tool, exception)
    record = ToolCallRecord(
        name=name,
        arguments=arguments,
        duration_ms=_elapsed_ms(started_ns),
        success=False,
        error_code=code,
    )
    return "", None, record, AgentError(code=code, message=message, tool=name)


def _error_details(tool: str, exception: Exception) -> tuple[str, str]:
    if isinstance(exception, ForecastIntegrationError):
        return exception.code, exception.message
    if isinstance(exception, MetricsSnapshotError):
        return exception.code, exception.message
    codes = {
        "knowledge": "KNOWLEDGE_SEARCH_FAILED",
        "metrics": "METRICS_UNAVAILABLE",
        "forecast": "FORECAST_UNAVAILABLE",
    }
    return codes[tool], str(exception)


def _elapsed_ms(started_ns: int) -> float:
    return round((time.perf_counter_ns() - started_ns) / 1_000_000, 3)
