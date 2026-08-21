import json

import pytest

from traffic_knowledge.evaluation.dataset import (
    EvaluationDatasetError,
    load_evaluation_questions,
)


def _question(question_id="q-1", **overrides):
    payload = {
        "id": question_id,
        "question": "PEMS04 包含多少个交通节点?",
        "category": "dataset_facts",
        "expected_answer_points": ["PEMS04 包含 307 个节点"],
        "relevant_chunk_ids": ["chunk-pems04"],
        "expected_tool": "knowledge",
    }
    payload.update(overrides)
    return payload


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_loads_valid_questions_as_immutable_contracts(tmp_path):
    path = tmp_path / "questions.jsonl"
    _write_jsonl(path, [_question()])

    questions = load_evaluation_questions(path)

    assert len(questions) == 1
    assert questions[0].id == "q-1"
    assert questions[0].expected_answer_points == ("PEMS04 包含 307 个节点",)
    assert questions[0].relevant_chunk_ids == ("chunk-pems04",)
    assert questions[0].expected_tool == "knowledge"


def test_rejects_duplicate_question_ids(tmp_path):
    path = tmp_path / "questions.jsonl"
    _write_jsonl(path, [_question(), _question()])

    with pytest.raises(EvaluationDatasetError, match="duplicate question id"):
        load_evaluation_questions(path)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"expected_answer_points": []}, "expected_answer_points"),
        ({"expected_answer_points": ["  "]}, "expected_answer_points"),
        ({"relevant_chunk_ids": []}, "relevant_chunk_ids"),
        ({"expected_tool": "shell"}, "expected_tool"),
    ],
)
def test_rejects_invalid_question_contracts(tmp_path, overrides, message):
    path = tmp_path / "questions.jsonl"
    _write_jsonl(path, [_question(**overrides)])

    with pytest.raises(EvaluationDatasetError, match=message):
        load_evaluation_questions(path)


def test_reports_json_line_number_for_invalid_json(tmp_path):
    path = tmp_path / "questions.jsonl"
    path.write_text(
        json.dumps(_question(), ensure_ascii=False) + "\nnot-json\n",
        encoding="utf-8",
    )

    with pytest.raises(EvaluationDatasetError, match="line 2"):
        load_evaluation_questions(path)
