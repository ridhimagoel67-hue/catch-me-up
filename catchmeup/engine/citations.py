
import re


REFUSAL_PHRASE = (
    "I couldn't find enough evidence in the content available to you."
)

CITATION_PATTERN = re.compile(r"\[([^\[\]]+)\]")


def extract_citation_ids(answer: str) -> list[str]:
    """Extract citation IDs written inside square brackets."""
    return CITATION_PATTERN.findall(answer)


def validate_citations(
    answer: str,
    supplied_chunk_ids: set[str],
    allow_uncited: bool = False,
) -> tuple[bool, list[str]]:
    """
    Validate citation IDs and optionally require a citation.

    Returns:
        A boolean indicating validity and a list of citation issues.
    """
    cited_ids = extract_citation_ids(answer)

    if not cited_ids and not allow_uncited:
        return False, ["Missing citation"]

    invalid_citations = [
        citation_id
        for citation_id in cited_ids
        if citation_id not in supplied_chunk_ids
    ]

    return len(invalid_citations) == 0, invalid_citations
