
from catchmeup.engine.citations import (
    REFUSAL_PHRASE,
    validate_citations,
)
from catchmeup.engine.workflow_state import CatchMeUpState


def validate_generated_answer(state: CatchMeUpState) -> dict:
    """
    Validate generated citations against the chunks supplied to the LLM.
    Invalid answers are marked for retry by the workflow.
    """
    answer = state.get("answer", "")
    supplied_chunk_ids = state.get("supplied_chunk_ids", set())

    if not isinstance(answer, str) or not answer.strip():
        return {
            "citation_valid": False,
            "citation_issues": ["Empty answer"],
            "status": "RETRY",
            "refusal_reason": "EMPTY_ANSWER",
        }

    if not isinstance(supplied_chunk_ids, set):
        supplied_chunk_ids = set()

    # Our standard refusal is allowed to omit a citation.
    is_refusal = answer.strip().startswith(REFUSAL_PHRASE)

    is_valid, issues = validate_citations(
        answer=answer,
        supplied_chunk_ids=supplied_chunk_ids,
        allow_uncited=is_refusal,
    )

    if not is_valid:
        return {
            "citation_valid": False,
            "citation_issues": issues,
            "status": "RETRY",
            "refusal_reason": "INVALID_CITATIONS",
        }

    return {
        "citation_valid": True,
        "citation_issues": [],
        "status": "VALIDATED",
        "refusal_reason": "",
    }
