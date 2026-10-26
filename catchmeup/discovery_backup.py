
import os
from urllib.parse import urlparse

import httpx
from dotenv import load_dotenv
from fastapi import HTTPException
from pydantic import BaseModel, Field

load_dotenv()

TAVILY_SEARCH_URL = "https://api.tavily.com/search"


# --------------------------------------------------
# 1. Request and response schemas
# --------------------------------------------------

class DiscoveryRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    content_type: str = Field(
        default="show",
        pattern="^(show|book|course|podcast|other)$",
    )
    max_results: int = Field(default=5, ge=1, le=10)


class SourceResult(BaseModel):
    title: str
    url: str
    score: float
    snippet: str
    source_type: str
    classification_reasons: list[str]


class DiscoveryResponse(BaseModel):
    query: str
    results: list[SourceResult]


# --------------------------------------------------
# 2. Content-specific search queries
# --------------------------------------------------

SEARCH_TERMS = {
    "show": (
        "official episode guide episode list "
        "season summaries episode order"
    ),
    "book": (
        "chapter list chapter summaries "
        "table of contents chapter order"
    ),
    "course": (
        "official course curriculum modules "
        "lessons syllabus learning materials"
    ),
    "podcast": (
        "official podcast episode list "
        "episode descriptions episode order"
    ),
    "other": (
        "official content guide structured "
        "index ordered content"
    ),
}


# --------------------------------------------------
# 3. Source classification
# --------------------------------------------------

def classify_source(
    title: str,
    url: str,
    snippet: str,
) -> tuple[str, list[str]]:
    """
    Classify a source using transparent heuristics.

    This is a preliminary classification, not a
    guarantee that the source is authoritative.
    """

    title_lower = title.lower()
    text = f"{title} {snippet}".lower()

    parsed_url = urlparse(url)
    host = (parsed_url.hostname or "").lower()

    if host.startswith("www."):
        host = host[4:]

    # A small list of known reference domains.
    reference_domains = (
        "wikipedia.org",
        "britannica.com",
    )

    # A small list of known educational providers.
    known_official_domains = (
        "developers.google.com",
        "learn.microsoft.com",
        "ocw.mit.edu",
    )

    # 1. Detect recap and commentary pages.
    recap_terms = (
        "recap",
        "review",
        "discussion",
        "reaction",
        "ending explained",
    )

    if any(term in title_lower for term in recap_terms):
        return (
            "recap_discussion",
            ["Title suggests recap or commentary"],
        )

    # 2. Detect structured content guides.
    structure_terms = (
        "episode list",
        "episode guide",
        "table of contents",
        "chapter list",
        "course curriculum",
        "lesson list",
        "syllabus",
        "module list",
    )

    if any(term in text for term in structure_terms):
        return (
            "structured_guide",
            ["Title or snippet suggests an ordered guide"],
        )

    # 3. Detect selected known reference sites.
    if any(
        host == domain or host.endswith("." + domain)
        for domain in reference_domains
    ):
        return (
            "reference",
            ["Domain matches a known reference site"],
        )

    # 4. Detect selected known learning providers.
    if any(
        host == domain or host.endswith("." + domain)
        for domain in known_official_domains
    ):
        return (
            "likely_official",
            ["Domain matches a known learning provider"],
        )

    # 5. Do not assume unknown sources are trustworthy.
    return (
        "unknown",
        ["Source type could not be established by these rules"],
    )


# --------------------------------------------------
# 4. Search and classify sources
# --------------------------------------------------

def discover_sources(
    request: DiscoveryRequest,
) -> DiscoveryResponse:

    api_key = os.getenv("TAVILY_API_KEY")

    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="Web discovery is not configured.",
        )

    # Build a query based on the content type.
    search_terms = SEARCH_TERMS[request.content_type]

    query = f'"{request.title}" {search_terms}'

    try:
        response = httpx.post(
            TAVILY_SEARCH_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
            },
            json={
                "query": query,
                "max_results": request.max_results,
                "search_depth": "basic",
                "topic": "general",
            },
            timeout=20.0,
        )

        response.raise_for_status()
        payload = response.json()

    except httpx.TimeoutException as exc:
        raise HTTPException(
            status_code=504,
            detail="Web search timed out. Please try again.",
        ) from exc

    except httpx.HTTPStatusError as exc:
        status_code = exc.response.status_code

        if status_code in (401, 403):
            raise HTTPException(
                status_code=502,
                detail="Tavily rejected the API credentials.",
            ) from exc

        if status_code == 429:
            raise HTTPException(
                status_code=503,
                detail="Tavily rate limit or quota reached.",
            ) from exc

        raise HTTPException(
            status_code=502,
            detail="The web search provider returned an error.",
        ) from exc

    except (httpx.RequestError, ValueError) as exc:
        raise HTTPException(
            status_code=502,
            detail="Could not complete web discovery.",
        ) from exc

    # Convert provider results into our own schema.
    results = []

    for item in payload.get("results", []):
        title = item.get("title")
        url = item.get("url")
        snippet = str(item.get("content", ""))

        # Skip malformed or non-HTTPS results.
        if not isinstance(title, str) or not title.strip():
            continue

        if not isinstance(url, str):
            continue

        parsed_url = urlparse(url)

        if (
            parsed_url.scheme != "https"
            or not parsed_url.hostname
            or parsed_url.username
            or parsed_url.password
        ):
            continue

        source_type, reasons = classify_source(
            title=title.strip(),
            url=url,
            snippet=snippet,
        )

        try:
            score = float(item.get("score", 0.0))
        except (TypeError, ValueError):
            score = 0.0

        results.append(
            SourceResult(
                title=title.strip(),
                url=url,
                score=score,
                snippet=snippet[:1500],
                source_type=source_type,
                classification_reasons=reasons,
            )
        )

    return DiscoveryResponse(
        query=query,
        results=results,
    )
