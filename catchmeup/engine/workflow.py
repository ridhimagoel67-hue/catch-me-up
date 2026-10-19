
from langgraph.graph import END, START, StateGraph

from catchmeup.engine.citations import REFUSAL_PHRASE
from catchmeup.engine.workflow_nodes import validate_request
from catchmeup.engine.workflow_retrieval import retrieve_evidence
from catchmeup.engine.workflow_safety import check_spoiler_safety
from catchmeup.engine.workflow_state import CatchMeUpState


def route_after_validation(state: CatchMeUpState) -> str:
    if state.get("status") == "READY":
        return "retrieve"
    return "stop"


def route_after_retrieval(state: CatchMeUpState) -> str:
    if state.get("status") == "ERROR":
        return "stop"

    if state.get("has_evidence"):
        return "check_safety"

    return "no_evidence"


def route_after_safety(state: CatchMeUpState) -> str:
    if state.get("status") != "READY":
        return "stop"
    return "continue"


def refuse_without_evidence(state: CatchMeUpState) -> dict:
    return {
        "answer": REFUSAL_PHRASE,
        "status": "REFUSED",
        "refusal_reason": "NO_EVIDENCE_RETRIEVED",
    }


def build_workflow():
    graph = StateGraph(CatchMeUpState)

    graph.add_node("validate_request", validate_request)
    graph.add_node("retrieve_evidence", retrieve_evidence)
    graph.add_node("check_spoiler_safety", check_spoiler_safety)
    graph.add_node("refuse_without_evidence", refuse_without_evidence)

    graph.add_edge(START, "validate_request")

    graph.add_conditional_edges(
        "validate_request",
        route_after_validation,
        {
            "retrieve": "retrieve_evidence",
            "stop": END,
        },
    )

    graph.add_conditional_edges(
        "retrieve_evidence",
        route_after_retrieval,
        {
            "check_safety": "check_spoiler_safety",
            "no_evidence": "refuse_without_evidence",
            "stop": END,
        },
    )

    graph.add_conditional_edges(
        "check_spoiler_safety",
        route_after_safety,
        {
            "stop": END,
            "continue": END,
        },
    )

    graph.add_edge("refuse_without_evidence", END)

    return graph.compile()

