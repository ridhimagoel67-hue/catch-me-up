from pathlib import Path

from catchmeup.engine.chunking import chunk_all
from catchmeup.engine.embedding import embed
from catchmeup.engine.retriever import Retriever
from scripts.test_rag import load_units


DATA_PATH = Path("data/processed/gossip-girl_s1.json")


def test_hybrid_retrieval_finds_direct_evidence_for_q02():
    units = load_units(DATA_PATH)
    chunks = chunk_all(units)
    retriever = Retriever(chunks, embed)

    results = retriever.search_hybrid(
        query="What is the situation between Dan and Jenny Humphrey's parents?",
        position=2,
        k=5,
    )

    result_ids = [chunk.chunk_id for chunk, _ in results]
    target = next(
        (chunk for chunk, _ in results if chunk.chunk_id == "s01e01-c01"),
        None,
    )

    assert target is not None, (
        "Direct evidence chunk s01e01-c01 was not found in the top five "
        f"hybrid results. Retrieved: {result_ids}"
    )
    assert "marriage is on a sabbatical" in target.text.lower()

