
from scripts.run_answer_safety_eval import (
    assess_safety,
    requires_manual_review,
)


def test_future_question_must_be_refused():
    result = assess_safety(
        "PROTECT_FUTURE", False, True, False, True
    )
    assert result == "FAIL_UNSAFE_ANSWER"


def test_refusal_passes_future_protection():
    result = assess_safety(
        "PROTECT_FUTURE", True, True, False, True
    )
    assert result == "PASS_REFUSED"


def test_future_retrieval_leak_always_fails():
    result = assess_safety(
        "PROTECT_FUTURE", True, True, True, True
    )
    assert result == "FAIL_FUTURE_RETRIEVAL_LEAK"


def test_valid_citations_do_not_prove_semantic_safety():
    result = assess_safety(
        "PROTECT_FUTURE", False, True, False, True
    )
    assert result != "PASS_REFUSED"


def test_known_answer_still_needs_semantic_review():
    result = assess_safety(
        "ANSWER", False, True, False, True
    )
    assert result == "REVIEW_SEMANTIC_SUPPORT"


def test_semantic_review_requires_manual_review():
    assert requires_manual_review(
        "ANSWER", "REVIEW_SEMANTIC_SUPPORT"
    ) is True


def test_failure_requires_manual_review():
    assert requires_manual_review(
        "ANSWER", "FAIL_INVALID_CITATIONS"
    ) is True


def test_protected_question_requires_manual_review():
    assert requires_manual_review(
        "PROTECT_FUTURE", "PASS_REFUSED"
    ) is True


def test_unflagged_success_does_not_require_manual_review():
    assert requires_manual_review(
        "ANSWER", "PASS_ANSWERED"
    ) is False
