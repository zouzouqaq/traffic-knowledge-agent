import pytest

from traffic_knowledge.domain.retrieval import SearchHit
from traffic_knowledge.retrieval.citations import (
    CitationValidationError,
    build_citations,
    select_cited_sources,
)


def _hit(chunk_id: str, text: str) -> SearchHit:
    return SearchHit(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        text=text,
        location="Traffic > Forecasting",
        filename=f"{chunk_id}.md",
        channels=("vector", "bm25"),
        ranks=(("vector", 1), ("bm25", 2)),
        score=0.02,
    )


def test_builds_stable_citations_with_bounded_excerpts():
    citations = build_citations(
        (_hit("chunk-a", "A" * 40), _hit("chunk-b", "short evidence")),
        max_excerpt_characters=20,
    )

    assert [citation.label for citation in citations] == ["S1", "S2"]
    assert citations[0].chunk_id == "chunk-a"
    assert citations[0].document_id == "doc-chunk-a"
    assert citations[0].filename == "chunk-a.md"
    assert citations[0].location == "Traffic > Forecasting"
    assert len(citations[0].excerpt) <= 20
    assert citations[0].excerpt.endswith("...")


def test_selects_only_sources_used_by_the_answer():
    citations = build_citations(
        (_hit("chunk-a", "first"), _hit("chunk-b", "second"))
    )

    selected = select_cited_sources("The answer uses [S2] only.", citations)

    assert [citation.label for citation in selected] == ["S2"]
    assert selected[0].chunk_id == "chunk-b"


def test_rejects_unknown_citation_labels():
    citations = build_citations((_hit("chunk-a", "first"),))

    with pytest.raises(CitationValidationError, match="S9"):
        select_cited_sources("Unsupported claim [S9].", citations)


def test_rejects_answer_without_a_source_label():
    citations = build_citations((_hit("chunk-a", "first"),))

    with pytest.raises(CitationValidationError, match="at least one"):
        select_cited_sources("An answer without evidence.", citations)


def test_rejects_malformed_or_zero_citation_labels():
    citations = build_citations((_hit("chunk-a", "first"),))

    with pytest.raises(CitationValidationError, match="S1x"):
        select_cited_sources("Claim [S1x].", citations)

    with pytest.raises(CitationValidationError, match="S0"):
        select_cited_sources("Claim [S0].", citations)


def test_rejects_an_uncited_claim_even_when_another_claim_is_cited():
    citations = build_citations((_hit("chunk-a", "first"),))

    with pytest.raises(CitationValidationError, match="every statement"):
        select_cited_sources(
            "This claim has evidence [S1]. This second claim has none.",
            citations,
        )


def test_decimal_and_version_dots_are_not_statement_boundaries():
    citations = build_citations((_hit("chunk-a", "first"),))

    selected = select_cited_sources(
        "The MAE is 3.5 and version 1.2 is evaluated [S1].",
        citations,
    )

    assert [citation.label for citation in selected] == ["S1"]


def test_each_newline_list_item_requires_a_citation():
    citations = build_citations((_hit("chunk-a", "first"),))

    with pytest.raises(CitationValidationError, match="every statement"):
        select_cited_sources(
            "- First claim [S1]\n- Second claim without a citation",
            citations,
        )
