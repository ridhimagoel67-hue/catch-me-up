
from catchmeup.engine.citations import REFUSAL_PHRASE
from catchmeup.engine.workflow_state import CatchMeUpState


def generate_answer(state: CatchMeUpState) -> dict:
    """
    Generate an evidence-grounded answer using the injected LLM chain.
    Citation validation happens in a separate node.
    """
    answer_chain = state.get("answer_chain")

    if answer_chain is None:
        return {
            "answer": REFUSAL_PHRASE,
            "status": "ERROR",
            "refusal_reason": "ANSWER_CHAIN_NOT_CONFIGURED",
        }

    if not state.get("has_evidence") or not state.get("evidence_text"):
        return {
            "answer": REFUSAL_PHRASE,
            "status": "REFUSED",
            "refusal_reason": "NO_EVIDENCE_RETRIEVED",
        }

    try:
        response = answer_chain.invoke(
            {
                "collection": state.get("collection_name", "content"),
                "unit_name": state.get("unit_name", "current unit"),
                "position": state["current_position"],
                "question": state["question"],
                "evidence": state["evidence_text"],
                "retry_feedback": state.get("retry_feedback", ""),
            }
        )

        answer = getattr(response, "content", None)

        if not isinstance(answer, str) or not answer.strip():
            return {
                "answer": REFUSAL_PHRASE,
                "status": "ERROR",
                "refusal_reason": "EMPTY_MODEL_RESPONSE",
            }

        return {
            "answer": answer.strip(),
            "status": "GENERATED",
            "refusal_reason": "",
        }

    except Exception:
        # Do not expose provider credentials or raw exception details.
        return {
            "answer": REFUSAL_PHRASE,
            "status": "ERROR",
            "refusal_reason": "ANSWER_GENERATION_FAILED",
        }
