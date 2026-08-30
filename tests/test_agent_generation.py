from pathlib import Path

from traffic_knowledge.application.agent_graph import AgentDependencies, build_agent_graph
from traffic_knowledge.application.grounded_answers import GroundedAnswerResult
from traffic_knowledge.application.question_answering import AnswerResult
from traffic_knowledge.domain.agent import AnswerGenerationMetadata
from traffic_knowledge.integrations.forecast_client import ForecastResult
from traffic_knowledge.integrations.metrics_snapshot import (
    ForecastHorizon,
    MetricsSnapshot,
    ModelMetrics,
)
from traffic_knowledge.retrieval.citations import Citation


class CombinedIntentModel:
    def classify(self, question):
        return "combined"


class QuestionAnsweringService:
    def answer(self, question):
        citation = Citation(
            label="S1",
            document_id="doc-1",
            filename="guide.md",
            location="paragraph 1",
            chunk_id="chunk-1",
            excerpt="GRU 使用历史交通流。",
        )
        return AnswerResult(
            answer="GRU 使用历史交通流 [S1]。",
            citations=(citation,),
            insufficient_evidence=False,
            elapsed_ms=1.0,
            evidence=(citation,),
        )


class MetricsRepository:
    def load(self, path):
        return MetricsSnapshot(
            schema_version="1.0",
            dataset="PEMS04",
            split="test",
            horizon=ForecastHorizon(steps=12, interval_minutes=5),
            created_at="2026-08-19T13:30:00+08:00",
            environment={"device": "cpu"},
            models=(ModelMetrics(name="gru", mae=27.1, rmse=41.0, mape=23.0),),
        )


class ForecastClient:
    def forecast(self, model, inputs):
        return ForecastResult(
            model=model,
            input_shape=(1, 2, 1, 1),
            output_shape=(1, 1, 1, 1),
            predictions=[[[[3.0]]]],
        )


class RecordingAnswerGenerator:
    def __init__(self):
        self.contexts = []

    def generate(self, context):
        self.contexts.append(context)
        return GroundedAnswerResult(
            answer="统一生成的回答 [S1]。",
            citations=context.knowledge.citations,
            generation=AnswerGenerationMetadata(
                answer_mode="deepseek",
                answer_model="deepseek-v4-flash",
                prompt_tokens=100,
                completion_tokens=20,
            ),
        )


def test_final_generator_receives_all_successful_tool_results():
    generator = RecordingAnswerGenerator()
    graph = build_agent_graph(
        AgentDependencies(
            intent_model=CombinedIntentModel(),
            qa_service=QuestionAnsweringService(),
            forecast_client=ForecastClient(),
            metrics_repository=MetricsRepository(),
            metrics_path=Path("metrics.json"),
            answer_generator=generator,
        )
    )

    response = graph.invoke(
        {
            "question": "综合分析",
            "forecast_model": "gru",
            "forecast_inputs": [[[[1.0]], [[2.0]]]],
        }
    )["response"]

    context = generator.contexts[0]
    assert context.knowledge is not None
    assert context.metrics is not None
    assert context.forecast is not None
    assert response.answer == "统一生成的回答 [S1]。"
    assert response.generation.answer_mode == "deepseek"
    assert [call.name for call in response.tool_calls] == [
        "search_traffic_knowledge",
        "get_model_metrics",
        "run_traffic_forecast",
    ]
