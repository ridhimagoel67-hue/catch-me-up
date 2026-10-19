
from catchmeup.engine.workflow_state import CatchMeUpState


def retrieve_evidence(state: CatchMeUpState) -> dict:
    """Retrieve evidence using an existing Retriever instance."""

    retriever = state.get("retriever")

    if retriever is None:
        return {
            "retrieved_chunks": [],
            "evidence_text": "",
            "has_evidence": False,
            "status": "ERROR",
            "refusal_reason": "RETRIEVER_NOT_CONFIGURED",
        }

    question = state["question"]
    position = state["current_position"]

    # Use the existing hybrid retrieval implementation.
    results = retriever.search_hybrid(
        query=question,
        position=position,
        k=5,
    )

    # Each result is a tuple: (chunk_object, retrieval_score).
    chunks = [chunk for chunk, score in results]

    
    evidence_text = "\n\n".join(
        f"Chunk ID: {chunk.chunk_id}\n"
        f"Position: {chunk.position}\n"
        f"Label: {chunk.label}\n"
        f"Text: {chunk.text}"
        for chunk in chunks
    )

    supplied_chunk_ids = {chunk.chunk_id for chunk in chunks}


    return {
        "supplied_chunk_ids": supplied_chunk_ids,
        "retrieved_chunks": chunks,
        "evidence_text": evidence_text,
        "has_evidence": len(chunks) > 0,
        "status": "READY",
        "refusal_reason": "",
    }
