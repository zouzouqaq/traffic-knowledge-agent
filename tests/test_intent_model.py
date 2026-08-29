from pathlib import Path

import pytest

from traffic_knowledge.api.dependencies import RuleBasedIntentModel
from traffic_knowledge.evaluation.dataset import load_evaluation_questions


def test_routes_checked_in_evaluation_questions_to_expected_tools():
    project_root = Path(__file__).resolve().parents[1]
    questions = load_evaluation_questions(project_root / "evaluation" / "questions.jsonl")
    model = RuleBasedIntentModel()

    mismatches = {
        question.id: (question.expected_tool, model.classify(question.question))
        for question in questions
        if model.classify(question.question) != question.expected_tool
    }

    assert mismatches == {}


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("请预测下一小时的交通流量", "forecast"),
        ("请预测未来 12 步交通流量", "forecast"),
        ("请综合预测下一小时并比较模型指标", "combined"),
    ],
)
def test_keeps_actionable_forecast_routes(question, expected):
    assert RuleBasedIntentModel().classify(question) == expected


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("未来 12 步交通流量是多少?", "forecast"),
        ("GRU 的 MAE 是多少?", "metrics"),
        ("对比两个模型", "metrics"),
        ("GRU 的 MAE 表示什么?", "knowledge"),
        ("交通预测中的时间依赖是什么?", "knowledge"),
        ("预测服务不可用时知识库还能工作吗?", "knowledge"),
    ],
)
def test_routes_common_phrasings_without_overfitting_fixture(question, expected):
    assert RuleBasedIntentModel().classify(question) == expected
