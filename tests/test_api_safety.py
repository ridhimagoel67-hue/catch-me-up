
from types import SimpleNamespace

from fastapi.testclient import TestClient

from catchmeup.api import create_app


class FakeRetriever:
    def search_hybrid(self, query, position, k=5):
        chunk = SimpleNamespace(
            chunk_id="s01e01-c01",
            position=1,
            label="Episode 1",
            text="Blair attends a party.",
        )
        return [(chunk, 0.9)] if position >= 1 else []


class FakeAnswerChain:
    def invoke(self, inputs):
        return SimpleNamespace(
            content="Blair attends a party. [s01e01-c01]"
        )


def make_app():
    return create_app(
        retriever=FakeRetriever(),
        answer_chain=FakeAnswerChain(),
        units=[
            {"position": 1},
            {"position": 2},
            {"position": 3},
        ],
    )


def test_client_cannot_supply_safety_metadata():
    with TestClient(make_app()) as client:
        response = client.post(
            "/ask",
            json={
                "question": "What happens next?",
                "current_position": 1,
                "expected_behavior": "PROTECT_FUTURE",
                "reveal_position": 3,
            },
        )

    assert response.status_code == 422


def test_client_cannot_supply_internal_dependencies():
    with TestClient(make_app()) as client:
        response = client.post(
            "/ask",
            json={
                "question": "What happens?",
                "current_position": 1,
                "retriever": "fake-client-retriever",
                "answer_chain": "client-controlled-chain",
            },
        )

    assert response.status_code == 422


def test_workflow_exception_returns_generic_error():
    app = make_app()

    class ExplodingWorkflow:
        def invoke(self, state):
            raise RuntimeError("internal-secret-detail")

    with TestClient(app) as client:
        app.state.workflow = ExplodingWorkflow()

        response = client.post(
            "/ask",
            json={
                "question": "What happened?",
                "current_position": 1,
            },
        )

    assert response.status_code == 500
    assert response.json()["detail"] == (
        "The request could not be processed."
    )
    assert "internal-secret-detail" not in response.text


def test_workflow_error_status_returns_service_unavailable():
    app = make_app()

    class ErrorWorkflow:
        def invoke(self, state):
            return {
                "status": "ERROR",
                "answer": "internal failure",
                "refusal_reason": "ANSWER_GENERATION_FAILED",
            }

    with TestClient(app) as client:
        app.state.workflow = ErrorWorkflow()

        response = client.post(
            "/ask",
            json={
                "question": "What happened?",
                "current_position": 1,
            },
        )

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "The answering workflow could not complete the request."
    )
    assert "internal failure" not in response.text
