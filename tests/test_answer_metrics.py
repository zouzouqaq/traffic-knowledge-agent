import pytest

from traffic_knowledge.evaluation.answer_metrics import (
    AnswerEvaluationCase,
    build_benchmark_agent_input,
    classify_tool_calls,
    compute_answer_metrics,
)


def test_computes_citation_tool_and_success_metrics():
    cases = (
        AnswerEvaluationCase("q1", "knowledge", "knowledge", ("a",), ("a",), True),
        AnswerEvaluationCase("q2", "metrics", "knowledge", ("b",), ("x",), True),
        AnswerEvaluationCase("q3", "knowledge", "knowledge", ("c",), (), False),
    )

    metrics = compute_answer_metrics(cases)

    assert metrics.case_count == 3
    assert metrics.citation_correctness == pytest.approx(0.5)
    assert metrics.tool_selection_accuracy == pytest.approx(2 / 3)
    assert metrics.tool_call_success_rate == pytest.approx(2 / 3)
    assert metrics.tool_confusion_matrix == {
        "knowledge": {"knowledge": 2},
        "metrics": {"knowledge": 1},
    }


def test_rejects_duplicate_case_ids_and_empty_input():
    case = AnswerEvaluationCase("q1", "knowledge", "knowledge", ("a",), ("a",), True)

    with pytest.raises(ValueError, match="at least one"):
        compute_answer_metrics(())
    with pytest.raises(ValueError, match="duplicate"):
        compute_answer_metrics((case, case))


def test_benchmark_input_keeps_misrouted_forecast_calls_measurable():
    state = build_benchmark_agent_input("模型的预测误差如何解释?")

    assert state["question"] == "模型的预测误差如何解释?"
    assert state["forecast_model"] == "gru"
    assert state["forecast_inputs"] == [[[[0.0]]]]


def test_classifies_the_complete_tool_call_sequence():
    assert classify_tool_calls(("search_traffic_knowledge",)) == "knowledge"
    assert classify_tool_calls(("get_model_metrics",)) == "metrics"
    assert classify_tool_calls(("run_traffic_forecast",)) == "forecast"
    assert classify_tool_calls(
        (
            "search_traffic_knowledge",
            "get_model_metrics",
            "run_traffic_forecast",
        )
    ) == "combined"
    assert classify_tool_calls(
        ("search_traffic_knowledge", "get_model_metrics")
    ) == "knowledge+metrics"


def test_citation_correctness_penalizes_wrong_and_missing_citations():
    cases = (
        AnswerEvaluationCase(
            "q1", "knowledge", "knowledge", ("a",), ("a", "wrong"), True
        ),
        AnswerEvaluationCase("q2", "knowledge", "knowledge", ("b",), (), True),
        AnswerEvaluationCase("q3", "metrics", "metrics", ("c",), (), True),
    )

    metrics = compute_answer_metrics(cases)

    assert metrics.citation_correctness == pytest.approx(0.25)
