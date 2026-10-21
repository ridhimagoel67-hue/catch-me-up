
import pytest
from pydantic import ValidationError

from catchmeup.api_schemas import AskRequest, AskResponse


def test_ask_request_uses_default_collection():
    request = AskRequest(
        question="What happened?",
        current_position=1,
    )

    assert request.collection_id == "gossip-girl-s1"


def test_ask_request_accepts_custom_collection():
    request = AskRequest(
        collection_id="my-book",
        question="What happened?",
        current_position=2,
    )

    assert request.collection_id == "my-book"


def test_ask_request_rejects_empty_question():
    with pytest.raises(ValidationError):
        AskRequest(question="", current_position=1)


def test_ask_request_rejects_negative_position():
    with pytest.raises(ValidationError):
        AskRequest(question="What happened?", current_position=-1)


def test_ask_request_rejects_unexpected_fields():
    with pytest.raises(ValidationError):
        AskRequest(
            question="What happened?",
            current_position=1,
            safety_metadata={"expected_behavior": "PROTECT_FUTURE"},
        )


def test_ask_response_applies_default_values():
    response = AskResponse(
        answer="I couldn't find enough evidence.",
        status="REFUSED",
    )

    assert response.citation_issues == []
    assert response.spoiler_protected is False
    assert response.safety_flags == []
    assert response.retry_count == 0
