
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

import catchmeup.api as api_module
from catchmeup.api import create_app


class FakeRetriever:
    def search_hybrid(self, query, position, k=5):
        return []


class FakeAnswerChain:
    def invoke(self, inputs):
        return SimpleNamespace(content="Test answer.")


class CapturingWorkflow:
    def __init__(self):
        self.states = []

    def invoke(self, state):
        self.states.append(state)
        return {
            "status": "VALIDATED",
            "answer": "Test answer. [test-chunk-1]",
            "citation_valid": True,
            "citation_issues": [],
            "refusal_reason": None,
            "spoiler_protected": False,
            "safety_flags": [],
            "retry_count": 0,
        }


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


def test_unknown_collection_returns_404():
    app = make_app()

    with TestClient(app) as client:
        response = client.post(
            "/ask",
            json={
                "collection_id": "does-not-exist",
                "question": "What happens?",
                "current_position": 1,
            },
        )

    assert response.status_code == 404
    assert response.json()["detail"] == (
        "The requested collection is not registered."
    )


def test_custom_collection_uses_its_own_runtime_and_metadata(monkeypatch):
    app = make_app()

    config = SimpleNamespace(
        collection_id="test-book",
        name="Test Book",
        unit_name="chapter",
        data_path=Path("test-book.json"),
    )

    custom_retriever = FakeRetriever()
    custom_answer_chain = FakeAnswerChain()

    custom_runtime = {
        "units": [{"position": 10}, {"position": 11}],
        "retriever": custom_retriever,
        "answer_chain": custom_answer_chain,
        "min_position": 10,
        "max_position": 11,
    }

    build_calls = []

    def fake_get_collection(collection_id):
        if collection_id == "test-book":
            return config
        raise KeyError(f"Unknown collection: {collection_id}")

    def fake_build_runtime(selected_config):
        build_calls.append(selected_config.collection_id)
        return custom_runtime

    monkeypatch.setattr(
        api_module, "get_collection", fake_get_collection
    )
    monkeypatch.setattr(
        api_module, "build_runtime", fake_build_runtime
    )

    workflow = CapturingWorkflow()

    with TestClient(app) as client:
        app.state.workflow = workflow

        response = client.post(
            "/ask",
            json={
                "collection_id": "test-book",
                "question": "Summarize this chapter.",
                "current_position": 10,
            },
        )

    assert response.status_code == 200
    assert build_calls == ["test-book"]
    assert len(workflow.states) == 1

    state = workflow.states[0]
    assert state["collection_name"] == "Test Book"
    assert state["unit_name"] == "chapter"
    assert state["retriever"] is custom_retriever
    assert state["answer_chain"] is custom_answer_chain
    assert state["current_position"] == 10


def test_custom_collection_runtime_is_reused(monkeypatch):
    app = make_app()

    config = SimpleNamespace(
        collection_id="test-book",
        name="Test Book",
        unit_name="chapter",
        data_path=Path("test-book.json"),
    )

    build_calls = []

    custom_runtime = {
        "units": [{"position": 10}, {"position": 11}],
        "retriever": FakeRetriever(),
        "answer_chain": FakeAnswerChain(),
        "min_position": 10,
        "max_position": 11,
    }

    def fake_get_collection(collection_id):
        if collection_id == "test-book":
            return config
        raise KeyError(f"Unknown collection: {collection_id}")

    def fake_build_runtime(selected_config):
        build_calls.append(selected_config.collection_id)
        return custom_runtime

    monkeypatch.setattr(
        api_module, "get_collection", fake_get_collection
    )
    monkeypatch.setattr(
        api_module, "build_runtime", fake_build_runtime
    )

    workflow = CapturingWorkflow()

    with TestClient(app) as client:
        app.state.workflow = workflow

        for _ in range(2):
            response = client.post(
                "/ask",
                json={
                    "collection_id": "test-book",
                    "question": "Summarize this chapter.",
                    "current_position": 10,
                },
            )
            assert response.status_code == 200

    assert build_calls == ["test-book"]
    assert len(workflow.states) == 2


def test_custom_collection_rejects_out_of_range_position(monkeypatch):
    app = make_app()

    config = SimpleNamespace(
        collection_id="test-book",
        name="Test Book",
        unit_name="chapter",
        data_path=Path("test-book.json"),
    )

    custom_runtime = {
        "units": [{"position": 10}, {"position": 11}],
        "retriever": FakeRetriever(),
        "answer_chain": FakeAnswerChain(),
        "min_position": 10,
        "max_position": 11,
    }

    monkeypatch.setattr(
        api_module,
        "get_collection",
        lambda collection_id: config,
    )
    monkeypatch.setattr(
        api_module,
        "build_runtime",
        lambda selected_config: custom_runtime,
    )

    with TestClient(app) as client:
        response = client.post(
            "/ask",
            json={
                "collection_id": "test-book",
                "question": "What happens?",
                "current_position": 1,
            },
        )

    assert response.status_code == 422


def test_missing_dataset_returns_503(monkeypatch):
    app = make_app()

    config = SimpleNamespace(
        collection_id="missing-book",
        name="Missing Book",
        unit_name="chapter",
        data_path=Path("missing-book.json"),
    )

    def fake_get_collection(collection_id):
        if collection_id == "missing-book":
            return config
        raise KeyError(f"Unknown collection: {collection_id}")

    def fake_build_runtime(selected_config):
        raise FileNotFoundError("Sensitive internal file path")

    monkeypatch.setattr(
        api_module, "get_collection", fake_get_collection
    )
    monkeypatch.setattr(
        api_module, "build_runtime", fake_build_runtime
    )

    with TestClient(app) as client:
        response = client.post(
            "/ask",
            json={
                "collection_id": "missing-book",
                "question": "Summarize this chapter.",
                "current_position": 10,
            },
        )

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "The requested collection's dataset is unavailable."
    )
    assert "Sensitive internal file path" not in response.text
    assert "missing-book" not in client.app.state.runtimes


def test_initialization_failure_returns_503_and_is_not_cached(monkeypatch):
    app = make_app()

    config = SimpleNamespace(
        collection_id="broken-book",
        name="Broken Book",
        unit_name="chapter",
        data_path=Path("broken-book.json"),
    )

    build_calls = []

    def fake_get_collection(collection_id):
        if collection_id == "broken-book":
            return config
        raise KeyError(f"Unknown collection: {collection_id}")

    def fake_build_runtime(selected_config):
        build_calls.append(selected_config.collection_id)
        raise RuntimeError("Sensitive initialization detail")

    monkeypatch.setattr(
        api_module, "get_collection", fake_get_collection
    )
    monkeypatch.setattr(
        api_module, "build_runtime", fake_build_runtime
    )

    with TestClient(app) as client:
        for _ in range(2):
            response = client.post(
                "/ask",
                json={
                    "collection_id": "broken-book",
                    "question": "Summarize this chapter.",
                    "current_position": 10,
                },
            )

            assert response.status_code == 503
            assert response.json()["detail"] == (
                "The requested collection could not be initialized."
            )
            assert "Sensitive initialization detail" not in response.text
            assert "broken-book" not in client.app.state.runtimes

    # Failed initialization should be retried, not cached as a valid runtime.
    assert build_calls == ["broken-book", "broken-book"]
