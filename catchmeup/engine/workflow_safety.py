
from catchmeup.engine.citations import REFUSAL_PHRASE
from catchmeup.engine.spoiler_guard import should_protect_future
from catchmeup.engine.workflow_state import CatchMeUpState


def check_spoiler_safety(state: CatchMeUpState) -> dict:
    """Apply the existing spoiler guard using trusted metadata."""

    expected_behavior = state.get("expected_behavior")
    current_position = state["current_position"]
    reveal_position = state.get("reveal_position")

    # Do not infer a future-reveal requirement from a position alone.
    if expected_behavior != "PROTECT_FUTURE":
        return {
            "spoiler_protected": False,
            "safety_flags": [],
            "status": "READY",
        }

    # Missing trusted metadata must not be silently treated as safe.
    if reveal_position is None:
        return {
            "spoiler_protected": False,
            "safety_flags": ["MISSING_REVEAL_METADATA"],
            "status": "SAFETY_REVIEW",
        }

    protected = should_protect_future(
        current_position,
        reveal_position,
    )

    if protected:
        return {
            "spoiler_protected": True,
            "answer": REFUSAL_PHRASE,
            "refusal_reason": "FUTURE_REVEAL_PROTECTED",
            "safety_flags": ["FUTURE_REVEAL_PROTECTED"],
            "status": "REFUSED",
        }

    return {
        "spoiler_protected": False,
        "safety_flags": [],
        "status": "READY",
    }
