from traffic_knowledge.application.grounded_answers import (
    EvidenceOnlyAnswerGenerator,
    GroundedAnswerContext,
    ResilientDeepSeekAnswerGenerator,
)
from traffic_knowledge.application.question_answering import (
    AnswerResult,
    QuestionAnsweringService,
)
from traffic_knowledge.domain.retrieval import SearchHit
from traffic_knowledge.integrations.deepseek import DeepSeekGeneration
from traffic_knowledge.retrieval.citations import Citation


def _citation(label: str, excerpt: str) -> Citation:
    return Citation(
        label=label,
        document_id=f"doc-{label}",
        filename=f"{label}.md",
        location="paragraph 1",
        chunk_id=f"chunk-{label}",
        excerpt=excerpt,
    )


def _knowledge_context() -> GroundedAnswerContext:
    evidence = (
        _citation("S1", "PEMS04 包含 307 个交通监测节点。"),
        _citation("S2", "每个样本使用 12 个历史时间步。"),
    )
    return GroundedAnswerContext(
        question="请概括 PEMS04 的节点和输入长度。",
        knowledge=AnswerResult(
            answer="PEMS04 包含 307 个节点 [S1]。",
            citations=(evidence[0],),
            insufficient_evidence=False,
            elapsed_ms=1.0,
            evidence=evidence,
        ),
    )


class RecordingDeepSeekClient:
    def __init__(self, answer: str):
        self.answer = answer
        self.system_prompt = ""
        self.user_prompt = ""

    def generate(self, system_prompt: str, user_prompt: str) -> DeepSeekGeneration:
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        return DeepSeekGeneration(
            content=self.answer,
            model="deepseek-v4-flash",
            prompt_tokens=100,
            completion_tokens=20,
            duration_ms=25.0,
        )


def test_deepseek_generator_uses_all_evidence_and_keeps_used_citations():
    client = RecordingDeepSeekClient(
        "PEMS04 包含 307 个节点 [S1]。输入包含 12 个历史步 [S2]。"
    )
    generator = ResilientDeepSeekAnswerGenerator(
        client=client,
        fallback=EvidenceOnlyAnswerGenerator(),
    )

    result = generator.generate(_knowledge_context())

    assert [citation.label for citation in result.citations] == ["S1", "S2"]
    assert result.generation.answer_mode == "deepseek"
    assert result.generation.answer_model == "deepseek-v4-flash"
    assert result.generation.llm_fallback is False
    assert '<evidence label="S2"' in client.user_prompt
    assert "untrusted evidence" in client.system_prompt.lower()


def test_invalid_deepseek_citation_falls_back_to_evidence_answer():
    client = RecordingDeepSeekClient("不存在的证据 [S99]。")
    result = ResilientDeepSeekAnswerGenerator(
        client=client,
        fallback=EvidenceOnlyAnswerGenerator(),
    ).generate(_knowledge_context())

    assert result.answer == "PEMS04 包含 307 个节点 [S1]。"
    assert result.generation.answer_mode == "evidence"
    assert result.generation.llm_fallback is True
    assert result.generation.llm_error_code == "LLM_CITATION_INVALID"
    assert "[S99]" not in result.answer


def test_question_answering_preserves_all_retrieved_evidence():
    hits = (
        SearchHit(
            chunk_id="first",
            document_id="doc-first",
            text="First evidence.",
            location="page:1",
            filename="first.pdf",
            channels=("vector",),
            ranks=(("vector", 1),),
            score=0.02,
        ),
        SearchHit(
            chunk_id="second",
            document_id="doc-second",
            text="Second evidence.",
            location="page:2",
            filename="second.pdf",
            channels=("vector",),
            ranks=(("vector", 2),),
            score=0.01,
        ),
    )

    class Retriever:
        def search(self, query, top_k):
            return hits[:top_k]

    class Model:
        def generate(self, system_prompt, user_prompt):
            return "Only the first source is cited [S1]."

    result = QuestionAnsweringService(Retriever(), Model()).answer("Question")

    assert [item.label for item in result.citations] == ["S1"]
    assert [item.label for item in result.evidence] == ["S1", "S2"]
