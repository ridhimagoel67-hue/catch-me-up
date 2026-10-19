
import re


REFUSAL_PHRASE = (
    "I couldn't find enough evidence in the content available to you."
)

# Accept both [chunk-id] and 【chunk-id】 citation formats.
CITATION_PATTERN = re.compile(
    r"\[([^\[\]]+)\]|【([^【】]+)】"
)


def extract_citation_ids(answer: str) -> list[str]:
    """Extract citation IDs from square or full-width brackets."""
    if not isinstance(answer, str):
        return []

    citation_ids = []

    for match in CITATION_PATTERN.finditer(answer):
        # One capture group is populated, depending on the format.
        citation_id = match.group(1) or match.group(2)
        citation_ids.append(citation_id.strip())

    return citation_ids


def validate_citations(
    answer: str,
    supplied_chunk_ids: set[str],
    allow_uncited: bool = False,
) -> tuple[bool, list[str]]:
    """
    Validate citations against IDs supplied to the model.

    Returns:
        (is_valid, issues)

    Refusal answers may omit citations when allow_uncited=True.
    Any extracted citation must still refer to a supplied chunk.
    """
    cited_ids = extract_citation_ids(answer)

    if not cited_ids and not allow_uncited:
        return False, ["Missing citation"]

    invalid_citations = [
        citation_id
        for citation_id in cited_ids
        if citation_id not in supplied_chunk_ids
    ]

    if invalid_citations:
        return False, invalid_citations

    return True, []
