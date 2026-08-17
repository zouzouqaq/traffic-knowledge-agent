from itertools import pairwise

import pytest

from traffic_knowledge.domain.document import ParsedDocument, SourceBlock
from traffic_knowledge.ingestion.chunking import chunk_document


def _parsed(text: str) -> ParsedDocument:
    return ParsedDocument(
        filename="guide.md",
        media_type="text/markdown",
        blocks=(SourceBlock(text=text, location="Traffic > Models", ordinal=0),),
    )


def test_chunks_respect_bounds_and_keep_bounded_overlap():
    parsed = _parsed(
        "GRU predicts short-term flow. "
        "Historical average is a baseline. "
        "MAE measures absolute error."
    )

    chunks = chunk_document("doc-1", parsed, max_characters=48, overlap_characters=10)

    assert len(chunks) >= 2
    assert all(len(chunk.text) <= 48 for chunk in chunks)
    assert all(chunk.document_id == "doc-1" for chunk in chunks)
    assert all(chunk.location == "Traffic > Models" for chunk in chunks)
    for previous, current in pairwise(chunks):
        shared = max(
            (size for size in range(1, 11) if previous.text[-size:] == current.text[:size]),
            default=0,
        )
        assert shared <= 10


def test_chunk_ids_are_stable_for_identical_input():
    parsed = _parsed("First sentence. Second sentence. Third sentence.")

    first = chunk_document("doc-1", parsed, 32, 5)
    second = chunk_document("doc-1", parsed, 32, 5)

    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
    assert [chunk.text for chunk in first] == [chunk.text for chunk in second]


def test_one_indivisible_sentence_may_exceed_bound():
    long_sentence = "A" * 80 + "."

    chunks = chunk_document("doc-1", _parsed(long_sentence), 32, 5)

    assert len(chunks) == 1
    assert chunks[0].text == long_sentence


@pytest.mark.parametrize(
    ("max_characters", "overlap_characters"),
    [(0, 0), (10, -1), (10, 10), (10, 11)],
)
def test_rejects_invalid_chunk_configuration(max_characters, overlap_characters):
    with pytest.raises(ValueError):
        chunk_document("doc-1", _parsed("Traffic flow."), max_characters, overlap_characters)
