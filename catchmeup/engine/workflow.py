
from langgraph.graph import END, START, StateGraph

from catchmeup.engine.citations import REFUSAL_PHRASE
from catchmeup.engine.workflow_nodes import validate_request
from catchmeup.engine.workflow_retrieval import retrieve_evidence
from catchmeup.engine.workflow_safety import check_spoiler_safety
from catchmeup.engine.workflow_generation import generate_answer
from catchmeup.engine.workflow_validation import validate_generated_answer
from catchmeup.engine.workflow_retry import prepare_retry
from catchmeup.engine.workflow_state import CatchMeUpState


def route_after_validation(state: CatchMeUpState) -> str:
    return "retrieve" if state.get("status") == "READY" else "stop"


def route_after_retrieval(state: CatchMeUpState) -> str:
    if state.get("status") == "ERROR":
        return "stop"
    if state.get("has_evidence"):
        return "check_safety"
    return "no_evidence"


def route_after_safety(state: CatchMeUpState) -> str:
    return "generate" if state.get("status") == "READY" else "stop"


def route_after_generation(state: CatchMeUpState) -> str:
    return "validate_citations" if state.get("status") == "GENERATED" else "stop"


def route_after_citations(state: CatchMeUpState) -> str:
    if state.get("status") == "VALIDATED":
        return "done"
    if state.get("status") == "RETRY":
        return "prepare_retry"
    return "stop"


def route_after_retry(state: CatchMeUpState) -> str:
    return "generate" if state.get("status") == "RETRY" else "stop"


def refuse_without_evidence(state: CatchMeUpState) -> dict:
    return {
        "answer": REFUSAL_PHRASE,
        "status": "REFUSED",
        "refusal_reason": "NO_EVIDENCE_RETRIEVED",
        "citation_valid": True,
        "citation_issues": [],
    }


def build_workflow():
    graph = StateGraph(CatchMeUpState)

    graph.add_node("validate_request", validate_request)
    graph.add_node("retrieve_evidence", retrieve_evidence)
    graph.add_node("check_spoiler_safety", check_spoiler_safety)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("validate_generated_answer", validate_generated_answer)
    graph.add_node("prepare_retry", prepare_retry)
    graph.add_node("refuse_without_evidence", refuse_without_evidence)

    graph.add_edge(START, "validate_request")

    graph.add_conditional_edges(
        "validate_request",
        route_after_validation,
        {"retrieve": "retrieve_evidence", "stop": END},
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
        {"generate": "generate_answer", "stop": END},
    )

    graph.add_conditional_edges(
        "generate_answer",
        route_after_generation,
        {"validate_citations": "validate_generated_answer", "stop": END},
    )

    graph.add_conditional_edges(
        "validate_generated_answer",
        route_after_citations,
        {
            "done": END,
            "prepare_retry": "prepare_retry",
            "stop": END,
        },
    )

    graph.add_conditional_edges(
        "prepare_retry",
        route_after_retry,
        {"generate": "generate_answer", "stop": END},
    )

    graph.add_edge("refuse_without_evidence", END)

    return graph.compile()


