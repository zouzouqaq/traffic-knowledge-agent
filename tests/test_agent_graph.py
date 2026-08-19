from pathlib import Path

import pytest

from traffic_knowledge.application.agent_graph import (
    TOOL_NAMES,
    AgentDependencies,
    build_agent_graph,
)
from traffic_knowledge.application.question_answering import AnswerResult
from traffic_knowledge.integrations.forecast_client import (
    ForecastIntegrationError,
    ForecastResult,
)
from traffic_knowledge.integrations.metrics_snapshot import (
    ForecastHorizon,
    MetricsSnapshot,
    MetricsSnapshotError,
    ModelMetrics,
)
from traffic_knowledge.retrieval.citations import Citation


class FixedIntentModel:
    def __init__(self, intent):
        self.intent = intent

    def classify(self, question):
        return self.intent


class FakeQuestionAnsweringService:
    def __init__(self):
        self.questions = []

    def answer(self, question):
        self.questions.append(question)
        return AnswerResult(
            answer="交通流预测需要历史观测数据 [S1]。",
            citations=(
                Citation(
                    label="S1",
                    document_id="doc-1",
                    filename="guide.md",
                    location="paragraph 1",
                    chunk_id="chunk-1",
                    excerpt="交通流预测需要历史观测数据。",
                ),
            ),
            insufficient_evidence=False,
            elapsed_ms=1.0,
        )


class FakeMetricsRepository:
    def __init__(self, error=None):
        self.paths = []
        self.error = error

    def load(self, path):
        self.paths.append(path)
        if self.error is not None:
            raise self.error
        return MetricsSnapshot(
            schema_version="1.0",
            dataset="PEMS04",
            split="test",
            horizon=ForecastHorizon(steps=12, interval_minutes=5),
            created_at="2026-08-19T13:30:00+08:00",
            environment={"device": "cpu"},
            models=(
                ModelMetrics(name="gru", mae=27.1, rmse=41.0, mape=23.0),
                ModelMetrics(
                    name="historical_average",
                    mae=26.5,
                    rmse=43.5,
                    mape=16.6,
                ),
            ),
        )


class FakeForecastClient:
    def __init__(self, error=None):
        self.calls = []
        self.error = error

    def forecast(self, model, inputs):
        self.calls.append((model, inputs))
        if self.error is not None:
            raise self.error
        return ForecastResult(
            model=model,
            input_shape=(1, 2, 1, 1),
            output_shape=(1, 1, 1, 1),
            predictions=[[[[3.0]]]],
        )


def _graph(
    intent,
    forecast_client=None,
    metrics_repository=None,
    max_tool_calls=3,
):
    dependencies = AgentDependencies(
        intent_model=FixedIntentModel(intent),
        qa_service=FakeQuestionAnsweringService(),
        forecast_client=forecast_client or FakeForecastClient(),
        metrics_repository=metrics_repository or FakeMetricsRepository(),
        metrics_path=Path("metrics.json"),
        max_tool_calls=max_tool_calls,
    )
    return build_agent_graph(dependencies), dependencies


@pytest.mark.parametrize(
    ("intent", "expected_tools"),
    [
        ("knowledge", ["search_traffic_knowledge"]),
        ("metrics", ["get_model_metrics"]),
        ("forecast", ["run_traffic_forecast"]),
        (
            "combined",
            [
                "search_traffic_knowledge",
                "get_model_metrics",
                "run_traffic_forecast",
            ],
        ),
    ],
)
def test_routes_to_only_the_bounded_tools_for_each_intent(intent, expected_tools):
    graph, _ = _graph(intent)

    state = graph.invoke(
        {
            "question": "请分析交通流预测",
            "forecast_model": "gru",
            "forecast_inputs": [[[[1.0]], [[2.0]]]],
        }
    )

    response = state["response"]
    assert [call.name for call in response.tool_calls] == expected_tools
    assert all(call.success for call in response.tool_calls)
    assert not response.partial
    assert response.answer
    if intent in {"knowledge", "combined"}:
        assert response.citations[0].label == "S1"


def test_forecast_failure_returns_partial_combined_answer():
    failure = ForecastIntegrationError("FORECAST_UNAVAILABLE", "service is down")
    graph, _ = _graph("combined", forecast_client=FakeForecastClient(failure))

    response = graph.invoke(
        {
            "question": "综合分析",
            "forecast_model": "gru",
            "forecast_inputs": [[[[1.0]], [[2.0]]]],
        }
    )["response"]

    assert response.partial
    assert response.errors[0].code == "FORECAST_UNAVAILABLE"
    assert [call.success for call in response.tool_calls] == [True, True, False]
    assert "PEMS04" in response.answer


def test_metrics_failure_returns_partial_combined_answer():
    repository = FakeMetricsRepository(MetricsSnapshotError("invalid snapshot"))
    graph, _ = _graph("combined", metrics_repository=repository)

    response = graph.invoke(
        {
            "question": "综合分析",
            "forecast_model": "gru",
            "forecast_inputs": [[[[1.0]], [[2.0]]]],
        }
    )["response"]

    assert response.partial
    assert response.errors[0].code == "METRICS_SCHEMA_INVALID"
    assert [call.success for call in response.tool_calls] == [True, False, True]


def test_forecast_answer_contains_a_numeric_prediction_summary():
    graph, _ = _graph("forecast")

    response = graph.invoke(
        {
            "question": "预测下一小时",
            "forecast_model": "gru",
            "forecast_inputs": [[[[1.0]], [[2.0]]]],
        }
    )["response"]

    assert "3.000" in response.answer


def test_combined_route_never_exceeds_tool_call_limit():
    graph, _ = _graph("combined", max_tool_calls=2)

    response = graph.invoke(
        {
            "question": "综合分析",
            "forecast_model": "gru",
            "forecast_inputs": [[[[1.0]], [[2.0]]]],
        }
    )["response"]

    assert len(response.tool_calls) == 2
    assert response.partial
    assert response.errors[-1].code == "AGENT_TOOL_LIMIT"


def test_forecast_route_validates_required_inputs():
    graph, _ = _graph("forecast")

    with pytest.raises(ValueError, match="forecast_inputs"):
        graph.invoke({"question": "预测下一小时", "forecast_model": "gru"})


def test_exposes_only_three_named_tools_and_no_code_execution_tool():
    assert TOOL_NAMES == (
        "search_traffic_knowledge",
        "get_model_metrics",
        "run_traffic_forecast",
    )
    assert all("exec" not in name and "code" not in name for name in TOOL_NAMES)
