
from catchmeup.engine.chunking import Chunk
from catchmeup.engine.retriever import Retriever
from catchmeup.engine.citations import (extract_citation_ids, validate_citations)


def fake_embed(texts):
    vectors = {
        "question": [1.0, 0.0],
        "relevant evidence": [1.0, 0.0],
        "irrelevant evidence": [0.0, 1.0],
    }
    return [vectors.get(text, [0.0, 1.0]) for text in texts]


def make_chunks():
    return [
        Chunk(
            chunk_id="unit-01-c01",
            unit_id="unit-01",
            position=1,
            label="Unit 1",
            text="relevant evidence",
        ),
        Chunk(
            chunk_id="unit-02-c01",
            unit_id="unit-02",
            position=2,
            label="Unit 2",
            text="irrelevant evidence",
        ),
        Chunk(
            chunk_id="unit-03-c01",
            unit_id="unit-03",
            position=3,
            label="Unit 3",
            text="relevant evidence",
        ),
    ]


def test_retriever_never_returns_future_content():
    retriever = Retriever(make_chunks(), fake_embed)

    hits = retriever.search("question", position=2, k=10)

    assert hits
    assert all(chunk.position <= 2 for chunk, score in hits)


def test_retriever_returns_no_content_at_position_zero():
    retriever = Retriever(make_chunks(), fake_embed)

    assert retriever.search("question", position=0) == []


def test_retriever_ranks_relevant_evidence_first():
    retriever = Retriever(make_chunks(), fake_embed)

    hits = retriever.search("question", position=3, k=3)

    assert hits[0][0].text == "relevant evidence"


def test_citation_ids_can_be_checked_against_supplied_evidence():
    supplied_ids = {"unit-01-c01", "unit-02-c01"}
    cited_ids = {"unit-01-c01"}

    assert cited_ids <= supplied_ids


def test_unknown_citation_is_detected():
    supplied_ids = {"unit-01-c01", "unit-02-c01"}
    cited_ids = {"unit-01-c01", "unit-99-c01"}

    assert not cited_ids <= supplied_ids



def test_extracts_citation_ids_from_answer():
    answer = (
        "Dan is Jenny's brother. "
        "[s01e01-c07] He attends school. [s01e01-c10]"
    )

    assert extract_citation_ids(answer) == [
        "s01e01-c07",
        "s01e01-c10",
    ]


def test_accepts_citations_present_in_supplied_evidence():
    answer = "Dan is Jenny's brother. [unit-01-c01]"
    supplied_ids = {"unit-01-c01", "unit-02-c01"}

    is_valid, invalid = validate_citations(answer, supplied_ids)

    assert is_valid is True
    assert invalid == []


def test_rejects_citations_missing_from_supplied_evidence():
    answer = "Dan is Jenny's brother. [unit-99-c01]"
    supplied_ids = {"unit-01-c01", "unit-02-c01"}

    is_valid, invalid = validate_citations(answer, supplied_ids)

    assert is_valid is False
    assert invalid == ["unit-99-c01"]

def test_rejects_answer_without_citations():
    answer = "Dan is Jenny's brother."
    supplied_ids = {"unit-01-c01"}

    is_valid, issues = validate_citations(answer, supplied_ids)

    assert is_valid is False
    assert issues == ["Missing citation"]


def test_allows_uncited_refusal_when_enabled():
    answer = (
        "I couldn't find enough evidence in the content "
        "available to you."
    )

    is_valid, issues = validate_citations(
        answer,
        supplied_chunk_ids={"unit-01-c01"},
        allow_uncited=True,
    )

    assert is_valid is True
    assert issues == []


def test_still_rejects_unknown_citations_when_uncited_allowed():
    answer = "Unsupported claim. [unit-99-c01]"

    is_valid, issues = validate_citations(
        answer,
        supplied_chunk_ids={"unit-01-c01"},
        allow_uncited=True,
    )

    assert is_valid is False
    assert issues == ["unit-99-c01"]

