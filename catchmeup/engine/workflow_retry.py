
from catchmeup.engine.citations import REFUSAL_PHRASE
from catchmeup.engine.workflow_state import CatchMeUpState


def prepare_retry(state: CatchMeUpState) -> dict:
    """
    Decide whether another generation attempt is allowed.
    If the limit is exhausted, return a safe refusal.
    """
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)

    # Defensive handling of invalid retry settings.
    if (
        not isinstance(retry_count, int)
        or isinstance(retry_count, bool)
        or retry_count < 0
    ):
        return {
            "answer": REFUSAL_PHRASE,
            "status": "REFUSED",
            "refusal_reason": "INVALID_RETRY_STATE",
        }

    if (
        not isinstance(max_retries, int)
        or isinstance(max_retries, bool)
        or max_retries < 0
    ):
        return {
            "answer": REFUSAL_PHRASE,
            "status": "REFUSED",
            "refusal_reason": "INVALID_RETRY_LIMIT",
        }

    if retry_count >= max_retries:
        return {
            "answer": REFUSAL_PHRASE,
            "status": "REFUSED",
            "refusal_reason": "CITATION_RETRIES_EXHAUSTED",
        }

    issues = state.get("citation_issues", [])
    feedback = (
        "Your previous answer failed citation validation. "
        "Correct the problem and answer using only the supplied evidence. "
        "Every citation must use an exact supplied chunk ID. "
        "Do not invent citation IDs. "
        f"Validation issues: {issues}"
    )

    return {
        "retry_count": retry_count + 1,
        "retry_feedback": feedback,
        "status": "RETRY",
        "refusal_reason": "INVALID_CITATIONS",
    }
