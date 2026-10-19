
from scripts.run_answer_safety_eval import expected_behavior
import pandas as pd


def make_row(question_type, asked_at, reveal_position):
    return pd.Series({
        "type": question_type,
        "asked_at": asked_at,
        "reveal_position": reveal_position,
    })


def test_future_question_requires_protection():
    assert expected_behavior(
        make_row("B", 6, 10)
    ) == "PROTECT_FUTURE"


def test_future_question_at_reveal_is_answerable_if_supported():
    assert expected_behavior(
        make_row("B", 10, 10)
    ) == "ANSWER_IF_SUPPORTED"


def test_known_fact_can_be_answered():
    assert expected_behavior(
        make_row("A", 6, 4)
    ) == "ANSWER"
