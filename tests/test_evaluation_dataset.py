import json
from collections import Counter
from pathlib import Path

import pytest

from traffic_knowledge.evaluation.dataset import (
    EvaluationDatasetError,
    load_evaluation_questions,
)
from traffic_knowledge.ingestion.chunking import chunk_document
from traffic_knowledge.ingestion.loaders import load_document
from traffic_knowledge.ingestion.repository import DocumentRepository
from traffic_knowledge.ingestion.service import sha256_file


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


def test_checked_in_evaluation_set_has_fifty_balanced_questions():
    project_root = Path(__file__).resolve().parents[1]

    questions = load_evaluation_questions(project_root / "evaluation" / "questions.jsonl")
    category_counts = Counter(question.category for question in questions)

    assert len(questions) == 50
    assert category_counts == {
        "dataset_facts": 10,
        "metric_interpretation": 10,
        "model_mechanisms": 10,
        "model_comparison": 10,
        "operational_guidance": 10,
    }


def test_checked_in_relevance_ids_match_freshly_rebuilt_corpus(tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    corpus_dir = project_root / "evaluation" / "corpus"
    repository = DocumentRepository(tmp_path / "metadata.sqlite3")
    repository.initialize()
    generated_chunk_ids = set()

    for source_path in sorted(corpus_dir.glob("*.md")):
        record = repository.begin_ingestion(sha256_file(source_path), source_path.name)
        chunks = chunk_document(
            record.document_id,
            load_document(source_path),
            max_characters=1000,
            overlap_characters=100,
        )
        generated_chunk_ids.update(chunk.chunk_id for chunk in chunks)

    questions = load_evaluation_questions(project_root / "evaluation" / "questions.jsonl")
    judged_chunk_ids = {
        chunk_id for question in questions for chunk_id in question.relevant_chunk_ids
    }

    assert len(generated_chunk_ids) == 50
    assert judged_chunk_ids == generated_chunk_ids
