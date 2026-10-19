
from typing import Any, TypedDict


class CatchMeUpState(TypedDict, total=False):
    # Injected dependencies
    retriever: Any
    answer_chain: Any

    # User request
    question: str
    current_position: int
    collection_name: str
    unit_name: str

    # Trusted benchmark safety metadata
    expected_behavior: str
    reveal_position: int | None
    spoiler_protected: bool

    # Retrieval results
    retrieved_chunks: list[Any]
    supplied_chunk_ids: set[str]
    evidence_text: str

    # Answer generation and citation validation
    answer: str
    citation_valid: bool
    citation_issues: list[str]

    # Workflow decisions
    has_evidence: bool
    refusal_reason: str
    safety_flags: list[str]

    # Bounded retry handling
    retry_count: int
    max_retries: int

    # Final workflow status
    status: str
    retry_feedback: str
