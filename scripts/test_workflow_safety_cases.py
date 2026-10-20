
from types import SimpleNamespace

import pytest

from catchmeup.engine.chunking import chunk_all
from catchmeup.engine.embedding import embed
from catchmeup.engine.retriever import Retriever
from catchmeup.engine.workflow import build_workflow
from scripts.test_rag import load_units


class CountingFakeChain:
    """Track LLM calls and simulate invalid citations."""

    def __init__(self, always_invalid=False):
        self.calls = 0
        self.always_invalid = always_invalid

    def invoke(self, inputs):
        self.calls += 1

        if self.always_invalid:
            return SimpleNamespace(
                content="Unsupported claim. [fake-chunk-99]"
            )

        return SimpleNamespace(
            content="A claim based on evidence. [fake-chunk-99]"
        )


@pytest.fixture(scope="module")
def workflow_context():
    units = load_units("data/processed/gossip-girl_s1.json")
    chunks = chunk_all(units)
    retriever = Retriever(chunks, embed)
    graph = build_workflow()
    return retriever, graph


def base_state(retriever, chain):
    return {
        "question": "Who is Serena?",
        "current_position": 7,
        "collection_name": "Gossip Girl",
        "unit_name": "episode",
        "expected_behavior": "ANSWER",
        "retriever": retriever,
        "answer_chain": chain,
        "retry_count": 0,
        "max_retries": 2,
    }


def test_protected_question_never_calls_llm(workflow_context):
    retriever, graph = workflow_context
    chain = CountingFakeChain()
    state = base_state(retriever, chain)
    state["expected_behavior"] = "PROTECT_FUTURE"
    state["reveal_position"] = 11

    result = graph.invoke(state)

    assert result["status"] == "REFUSED"
    assert result["spoiler_protected"] is True
    assert result["answer"] == (
        "I couldn't find enough evidence in the content available to you."
    )
    assert chain.calls == 0


def test_missing_reveal_metadata_fails_closed(workflow_context):
    retriever, graph = workflow_context
    chain = CountingFakeChain()
    state = base_state(retriever, chain)
    state["expected_behavior"] = "PROTECT_FUTURE"

    result = graph.invoke(state)

    assert result["status"] == "REFUSED"
    assert result["refusal_reason"] == "MISSING_REVEAL_METADATA"
    assert result["answer"] == (
        "I couldn't find enough evidence in the content available to you."
    )
    assert chain.calls == 0


def test_exhausted_retries_return_refusal(workflow_context):
    retriever, graph = workflow_context
    chain = CountingFakeChain(always_invalid=True)
    state = base_state(retriever, chain)

    result = graph.invoke(state)

    assert result["status"] == "REFUSED"
    assert result["refusal_reason"] == "CITATION_RETRIES_EXHAUSTED"
    assert result["answer"] == (
        "I couldn't find enough evidence in the content available to you."
    )
    assert chain.calls == 3
    assert result["retry_count"] == 2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
