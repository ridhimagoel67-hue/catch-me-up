
"""Extract substantive source content using Tavily."""

import os
from urllib.parse import urlparse

import httpx
from dotenv import load_dotenv
from fastapi import HTTPException
from pydantic import BaseModel, Field

load_dotenv()

TAVILY_EXTRACT_URL = "https://api.tavily.com/extract"
MAX_CONTENT_CHARACTERS = 50_000


class ExtractionRequest(BaseModel):
    url: str = Field(min_length=1, max_length=2048)


class ExtractionResponse(BaseModel):
    url: str
    status: str
    content: str
    character_count: int
    truncated: bool


def validate_source_url(url: str) -> str:
    """Accept HTTPS URLs while rejecting obvious local hostnames."""
    parsed = urlparse(url.strip())

    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
    ):
        raise HTTPException(
            status_code=422,
            detail="A valid HTTPS source URL is required.",
        )

    hostname = parsed.hostname.lower()

    if (
        hostname == "localhost"
        or hostname.endswith((".localhost", ".local", ".internal"))
    ):
        raise HTTPException(
            status_code=422,
            detail="Local network URLs are not allowed.",
        )

    return url.strip()


def extract_source(request: ExtractionRequest) -> ExtractionResponse:
    """Extract deeper content and combine successful results."""
    api_key = os.getenv("TAVILY_API_KEY")

    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="Web extraction is not configured.",
        )

    url = validate_source_url(request.url)

    try:
        response = httpx.post(
            TAVILY_EXTRACT_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "urls": [url],
                "extract_depth": "advanced",
                "format": "markdown",
            },
            timeout=45.0,
        )
        response.raise_for_status()
        payload = response.json()

    except httpx.TimeoutException as exc:
        raise HTTPException(
            status_code=504,
            detail="Source extraction timed out.",
        ) from exc

    except httpx.HTTPStatusError as exc:
        if exc.response.status_code in (401, 403):
            detail = "Tavily rejected the API credentials."
            status_code = 502
        elif exc.response.status_code == 429:
            detail = "Tavily rate limit or quota reached."
            status_code = 503
        else:
            detail = "The extraction provider returned an error."
            status_code = 502

        raise HTTPException(status_code=status_code, detail=detail) from exc

    except (httpx.RequestError, ValueError) as exc:
        raise HTTPException(
            status_code=502,
            detail="Could not extract the selected source.",
        ) from exc

    if not isinstance(payload, dict):
        raise HTTPException(
            status_code=502,
            detail="The extraction provider returned an invalid response.",
        )

    results = payload.get("results", [])
    if not isinstance(results, list):
        results = []

    # Keep every successful non-empty result instead of silently
    # discarding content after the first result.
    sections = []
    source_urls = []

    for result in results:
        if not isinstance(result, dict):
            continue

        raw_content = result.get("raw_content")
        if not isinstance(raw_content, str) or not raw_content.strip():
            continue

        result_url = result.get("url") or url
        source_urls.append(result_url)
        sections.append(
            f"Source: {result_url}\n\n{raw_content.strip()}"
        )

    if not sections:
        failed_results = payload.get("failed_results", [])
        if failed_results:
            raise HTTPException(
                status_code=422,
                detail=(
                    "Tavily could not extract this page. "
                    "Try a different source with readable text."
                ),
            )

        raise HTTPException(
            status_code=422,
            detail=(
                "No readable content was returned. "
                "Try a source containing summaries or full text, "
                "not just a list of titles."
            ),
        )

    content = "\n\n---\n\n".join(sections)
    truncated = len(content) > MAX_CONTENT_CHARACTERS
    content = content[:MAX_CONTENT_CHARACTERS]

    return ExtractionResponse(
        url=source_urls[0] if source_urls else url,
        status="EXTRACTED",
        content=content,
        character_count=len(content),
        truncated=truncated,
    )

