
from catchmeup.engine.workflow_state import CatchMeUpState


def validate_request(state: CatchMeUpState) -> dict:
    """Validate the basic inputs before retrieval begins."""

    question = state.get("question", "").strip()
    position = state.get("current_position")

    if not question:
        return {
            "status": "REFUSED",
            "answer": (
                "I couldn't find enough evidence in the content available to you."
            ),
            "refusal_reason": "EMPTY_QUESTION",
        }

    if (
        not isinstance(position, int)
        or isinstance(position, bool)
        or position < 0
    ):
        return {
            "status": "REFUSED",
            "answer": (
                "I couldn't find enough evidence in the content available to you."
            ),
            "refusal_reason": "INVALID_POSITION",
        }

    return {
        "question": question,
        "status": "READY",
        "refusal_reason": "",
    }
