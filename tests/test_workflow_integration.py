
from types import SimpleNamespace

from catchmeup.engine.chunking import chunk_all
from catchmeup.engine.embedding import embed
from catchmeup.engine.retriever import Retriever
from catchmeup.engine.workflow import build_workflow
from scripts.test_rag import load_units


class FakeChain:
    """Return one invalid citation, then a corrected one."""

    def __init__(self):
        self.calls = 0
        self.feedback_seen = []

    def invoke(self, inputs):
        self.calls += 1
        self.feedback_seen.append(inputs.get("retry_feedback", ""))

        if self.calls == 1:
            return SimpleNamespace(
                content="Serena returns home. [fake-chunk-99]"
            )

        # Use a real chunk ID from the evidence supplied to this call.
        evidence = inputs["evidence"]
        chunk_id = evidence.split("Chunk ID: ", 1)[1].splitlines()[0]

        return SimpleNamespace(
            content=f"Based on the available passage. [{chunk_id}]"
        )


def main():
    units = load_units("data/processed/gossip-girl_s1.json")
    chunks = chunk_all(units)
    retriever = Retriever(chunks, embed)
    fake_chain = FakeChain()
    graph = build_workflow()

    result = graph.invoke({
        "question": "Who is Serena?",
        "current_position": 7,
        "collection_name": "Gossip Girl",
        "unit_name": "episode",
        "expected_behavior": "ANSWER",
        "retriever": retriever,
        "answer_chain": fake_chain,
        "retry_count": 0,
        "max_retries": 2,
    })

    print("STATUS:", result.get("status"))
    print("ANSWER:", result.get("answer"))
    print("CITATION VALID:", result.get("citation_valid"))
    print("RETRY COUNT:", result.get("retry_count"))
    print("LLM CALLS:", fake_chain.calls)
    print("RETRY FEEDBACK PROVIDED:", bool(
        fake_chain.feedback_seen[-1]
    ))

    assert result.get("status") == "VALIDATED"
    assert result.get("citation_valid") is True
    assert result.get("retry_count") == 1
    assert fake_chain.calls == 2
    assert bool(fake_chain.feedback_seen[-1])
    assert "[fake-chunk-99]" not in result.get("answer", "")

    print("\nPASS: Invalid citation was rejected and corrected.")



def test_invalid_citation_is_rejected_and_corrected():
    units = load_units("data/processed/gossip-girl_s1.json")
    chunks = chunk_all(units)
    retriever = Retriever(chunks, embed)
    fake_chain = FakeChain()
    graph = build_workflow()

    result = graph.invoke({
        "question": "Who is Serena?",
        "current_position": 7,
        "collection_name": "Gossip Girl",
        "unit_name": "episode",
        "expected_behavior": "ANSWER",
        "retriever": retriever,
        "answer_chain": fake_chain,
        "retry_count": 0,
        "max_retries": 2,
    })

    assert result.get("status") == "VALIDATED"
    assert result.get("citation_valid") is True
    assert result.get("retry_count") == 1
    assert fake_chain.calls == 2
    assert bool(fake_chain.feedback_seen[-1])
    assert "[fake-chunk-99]" not in result.get("answer", "")

