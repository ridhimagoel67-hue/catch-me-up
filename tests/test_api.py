
from types import SimpleNamespace

from fastapi.testclient import TestClient

from catchmeup.api import create_app


class FakeRetriever:
    def __init__(self):
        self.chunk = SimpleNamespace(
            chunk_id="s01e01-c01",
            position=1,
            label="Episode 1",
            text="Blair attends a party.",
        )

    def search_hybrid(self, query, position, k=5):
        if self.chunk.position <= position:
            return [(self.chunk, 0.9)]
        return []


class FakeAnswerChain:
    def invoke(self, inputs):
        return SimpleNamespace(
            content="Blair attends a party. [s01e01-c01]"
        )


def make_client():
    units = [
        {"position": 1},
        {"position": 2},
        {"position": 3},
    ]

    app = create_app(
        retriever=FakeRetriever(),
        answer_chain=FakeAnswerChain(),
        units=units,
    )

    return TestClient(app)


def test_health_endpoint():
    with make_client() as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ask_returns_validated_answer():
    with make_client() as client:
        response = client.post(
            "/ask",
            json={
                "question": "Where does Blair go?",
                "current_position": 1,
            },
        )

    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "VALIDATED"
    assert body["citation_valid"] is True
    assert "s01e01-c01" in body["answer"]


def test_negative_position_is_rejected():
    with make_client() as client:
        response = client.post(
            "/ask",
            json={
                "question": "What happened?",
                "current_position": -1,
            },
        )

    assert response.status_code == 422


def test_position_outside_dataset_is_rejected():
    with make_client() as client:
        response = client.post(
            "/ask",
            json={
                "question": "What happened?",
                "current_position": 99,
            },
        )

    assert response.status_code == 422


def test_empty_question_is_rejected():
    with make_client() as client:
        response = client.post(
            "/ask",
            json={
                "question": "",
                "current_position": 1,
            },
        )

    assert response.status_code == 422


def test_no_evidence_returns_refusal():
    class EmptyRetriever:
        def search_hybrid(self, query, position, k=5):
            return []

    units = [{"position": 1}, {"position": 2}]

    app = create_app(
        retriever=EmptyRetriever(),
        answer_chain=FakeAnswerChain(),
        units=units,
    )

    with TestClient(app) as client:
        response = client.post(
            "/ask",
            json={
                "question": "An unsupported question",
                "current_position": 1,
            },
        )

    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "REFUSED"
    assert body["refusal_reason"] == "NO_EVIDENCE_RETRIEVED"
    assert "couldn't find enough evidence" in body["answer"]
