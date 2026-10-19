
from catchmeup.engine.citations import REFUSAL_PHRASE


def should_protect_future(
    asked_at: int,
    reveal_position: int,
) -> bool:
    """Return True when a known outcome has not yet been revealed."""
    return int(asked_at) < int(reveal_position)


def apply_spoiler_guard(
    answer: str,
    asked_at: int,
    reveal_position: int,
) -> str:
    """
    Protect an annotated future outcome.

    The reveal position must come from trusted question metadata,
    not from the model's answer.
    """
    if should_protect_future(asked_at, reveal_position):
        return REFUSAL_PHRASE

    return answer
