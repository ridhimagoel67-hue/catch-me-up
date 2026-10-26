
import os
from urllib.parse import urlparse

import httpx
from dotenv import load_dotenv
from fastapi import HTTPException
from pydantic import BaseModel, Field

load_dotenv()

TAVILY_SEARCH_URL = "https://api.tavily.com/search"

SEARCH_TERMS = {
    "show": (
        "official episode guide episode list "
        "season summaries episode order"
    ),
    "book": (
        "chapter list table of contents "
        "chapter order chapter summaries"
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
        "official content guide structured index "
        "ordered content"
    ),
}


STRUCTURE_TERMS = {
    "show": (
        "episode list",
        "list of episodes",
        "episode guide",
        "episode order",
        "season guide",
        "season summaries",
        "episode summaries",
    ),
    "book": (
        "table of contents",
        "chapter list",
        "chapter order",
        "list of chapters",
    ),
    "course": (
        "course curriculum",
        "course syllabus",
        "lesson list",
        "course modules",
        "learning path",
    ),
    "podcast": (
        "episode list",
        "list of episodes",
        "episode guide",
        "episode order",
    ),
    "other": (
        "table of contents",
        "content index",
        "structured guide",
    ),
}


RECAP_TERMS = (
    "recap",
    "review",
    "discussion",
    "reaction",
    "ending explained",
)

REFERENCE_DOMAINS = (
    "wikipedia.org",
    "britannica.com",
)

KNOWN_LEARNING_DOMAINS = (
    "developers.google.com",
    "learn.microsoft.com",
    "ocw.mit.edu",
)


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
    ranking_score: float
    ranking_reasons: list[str]


class DiscoveryResponse(BaseModel):
    query: str
    results: list[SourceResult]


def get_host(url: str) -> str:
    """Return a normalized hostname."""
    host = urlparse(url).hostname or ""
    return host.lower().removeprefix("www.")


def domain_matches(host: str, domains: tuple[str, ...]) -> bool:
    """Match a domain without accepting lookalike suffixes."""
    return any(
        host == domain or host.endswith("." + domain)
        for domain in domains
    )




def classify_source(
    title: str,
    url: str,
    snippet: str,
    content_type: str = "show",
) -> tuple[str, list[str]]:
    """Classify a source using task-specific text and URL signals."""

    title_lower = title.lower()
    text = f"{title} {snippet}".lower()
    host = get_host(url)
    path = urlparse(url).path.lower()

    # 1. Recaps and commentary take priority.
    if any(term in title_lower for term in RECAP_TERMS):
        return (
            "recap_discussion",
            ["Title suggests recap or commentary"],
        )

    # 2. Recognize structured content in the title, snippet,
    #    and URL path. This check happens before reference domains.
    structure_terms = STRUCTURE_TERMS[content_type]

    if any(term in title_lower for term in structure_terms):
        return (
            "structured_guide",
            ["Title contains a content-type-specific guide indicator"],
        )

    if any(term in text for term in structure_terms):
        return (
            "structured_guide",
            ["Text contains a content-type-specific guide indicator"],
        )

    # Additional patterns for episode summaries that don't use
    # the exact phrase "episode summaries" in the page title.
    if content_type == "show":
        has_episode_number = any(
            marker in text
            for marker in ("episode 1", "episode 10", "episode 101")
        )
        has_episode_path = any(
            marker in path
            for marker in ("episode", "season", "episoderesources")
        )
        has_summary_signal = any(
            marker in text
            for marker in ("summary", "summaries", "pilot")
        )

        if (
            has_episode_number
            and (has_episode_path or has_summary_signal)
        ):
            return (
                "structured_guide",
                ["Episode numbering and summary indicators detected"],
            )

    # 3. Only classify as a reference if no stronger structure
    
    if domain_matches(host, REFERENCE_DOMAINS):
        # Wikipedia episode-list pages are useful for ordering.
        if (
            content_type == "show"
            and "wikipedia.org" in host
            and "list_of_" in path
            and "episodes" in path
        ):
            return (
                "structured_guide",
                ["Wikipedia URL identifies an episode-list page"],
            )

        return (
            "reference",
            ["Domain matches a known reference site"],
        )

    # 4. Known learning-provider domains.
    if domain_matches(host, KNOWN_LEARNING_DOMAINS):
        return (
            "likely_official",
            ["Domain matches a known learning provider"],
        )

    return (
        "unknown",
        ["Source type could not be established by these rules"],
    )





def rank_source(
    title: str,
    url: str,
    snippet: str,
    tavily_score: float,
    content_type: str,
) -> tuple[float, list[str]]:
    """
    Rank candidates using provider relevance and
    task-specific signals. The score is heuristic,
    not a probability of correctness.
    """

    score = max(0.0, min(1.0, tavily_score))
    reasons = [f"Tavily relevance: {score:.3f}"]

    source_type, _ = classify_source(
        title=title,
        url=url,
        snippet=snippet,
        content_type=content_type,
    )

    bonuses = {
        "structured_guide": 0.12,
        "likely_official": 0.08,
        "reference": 0.04,
        "unknown": 0.0,
        "recap_discussion": -0.15,
    }

    bonus = bonuses[source_type]
    score += bonus

    if bonus > 0:
        reasons.append(
            f"{source_type} bonus: +{bonus:.2f}"
        )
    elif bonus < 0:
        reasons.append(
            f"{source_type} penalty: {bonus:.2f}"
        )

    # Only reward specific guide phrases in the title.
    # Generic words like "episodes" are not enough.
    title_lower = title.lower()

    title_terms = {
        "show": (
            "episode guide",
            "episode list",
            "list of episodes",
            "season summaries",
            "episode summaries",
        ),
        "book": (
            "table of contents",
            "chapter list",
            "list of chapters",
        ),
        "course": (
            "course curriculum",
            "course syllabus",
            "lesson list",
            "learning path",
        ),
        "podcast": (
            "episode guide",
            "episode list",
            "list of episodes",
        ),
        "other": (
            "table of contents",
            "content index",
            "structured guide",
        ),
    }

    if any(
        term in title_lower
        for term in title_terms[content_type]
    ):
        score += 0.05
        reasons.append(
            "Specific title phrase matches content type: +0.05"
        )

    # Do not cap at 1.0. A score above 1 is allowed
    # because this is a ranking score, not a probability.
    return round(max(0.0, score), 4), reasons



def discover_sources(
    request: DiscoveryRequest,
) -> DiscoveryResponse:
    api_key = os.getenv("TAVILY_API_KEY")

    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="Web discovery is not configured.",
        )

    query = (
        f'"{request.title.strip()}" '
        f'{SEARCH_TERMS[request.content_type]}'
    )

    try:
        response = httpx.post(
            TAVILY_SEARCH_URL,
            headers={"Authorization": f"Bearer {api_key}"},
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

    results = []

    for item in payload.get("results", []):
        title = item.get("title")
        url = item.get("url")
        snippet = str(item.get("content", ""))

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

        try:
            tavily_score = float(item.get("score", 0.0))
        except (TypeError, ValueError):
            tavily_score = 0.0

        # Ignore malformed scores rather than allowing them
        # to corrupt the ranking.
        if not 0.0 <= tavily_score <= 1.0:
            tavily_score = 0.0

        title = title.strip()
        snippet = snippet[:1500]

        source_type, classification_reasons = classify_source(
            title=title,
            url=url,
            snippet=snippet,
            content_type=request.content_type,
        )

        ranking_score, ranking_reasons = rank_source(
            title=title,
            url=url,
            snippet=snippet,
            tavily_score=tavily_score,
            content_type=request.content_type,
        )

        results.append(
            SourceResult(
                title=title,
                url=url,
                score=round(tavily_score, 4),
                snippet=snippet,
                source_type=source_type,
                classification_reasons=classification_reasons,
                ranking_score=ranking_score,
                ranking_reasons=ranking_reasons,
            )
        )

    # Sort by our computed score, highest first.
    results.sort(
        key=lambda result: result.ranking_score,
        reverse=True,
    )

    return DiscoveryResponse(
        query=query,
        results=results,
    )
