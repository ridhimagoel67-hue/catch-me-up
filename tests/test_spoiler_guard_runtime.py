
from catchmeup.engine.citations import REFUSAL_PHRASE
from catchmeup.engine.spoiler_guard import (
    apply_spoiler_guard,
    should_protect_future,
)


def test_protects_outcome_before_reveal():
    assert should_protect_future(6, 10) is True


def test_allows_evaluation_at_reveal_position():
    assert should_protect_future(10, 10) is False


def test_replaces_answer_with_refusal_before_reveal():
    answer = "Yes, Lily eventually approves."

    result = apply_spoiler_guard(
        answer=answer,
        asked_at=6,
        reveal_position=10,
    )

    assert result == REFUSAL_PHRASE


def test_preserves_answer_at_reveal_position():
    answer = "The supplied evidence supports the answer."

    result = apply_spoiler_guard(
        answer=answer,
        asked_at=10,
        reveal_position=10,
    )

    assert result == answer


def test_does_not_change_answer_after_reveal():
    answer = "The supplied evidence supports the answer."

    result = apply_spoiler_guard(
        answer=answer,
        asked_at=12,
        reveal_position=10,
    )

    assert result == answer
