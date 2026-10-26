
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
    """Accept only public-looking HTTPS URLs."""

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

    # Reject localhost and obvious local-network hostnames.
    if (
        hostname == "localhost"
        or hostname.endswith(".localhost")
        or hostname.endswith(".local")
        or hostname.endswith(".internal")
    ):
        raise HTTPException(
            status_code=422,
            detail="Local network URLs are not allowed.",
        )

    return url.strip()


def extract_source(request: ExtractionRequest) -> ExtractionResponse:
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
            headers={
                "Authorization": f"Bearer {api_key}",
            },
            json={
                "urls": [url],
                "extract_depth": "basic",
                "format": "text",
            },
            timeout=30.0,
        )

        response.raise_for_status()
        payload = response.json()

    except httpx.TimeoutException as exc:
        raise HTTPException(
            status_code=504,
            detail="Source extraction timed out.",
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
            detail="The extraction provider returned an error.",
        ) from exc

    except (httpx.RequestError, ValueError) as exc:
        raise HTTPException(
            status_code=502,
            detail="Could not extract the selected source.",
        ) from exc

    failed_results = payload.get("failed_results", [])

    if failed_results:
        raise HTTPException(
            status_code=422,
            detail="Tavily could not extract the selected source.",
        )

    results = payload.get("results", [])

    if not results:
        raise HTTPException(
            status_code=422,
            detail="No extractable content was returned.",
        )

    extracted = results[0]
    content = extracted.get("raw_content")

    if not isinstance(content, str) or not content.strip():
        raise HTTPException(
            status_code=422,
            detail="The selected source returned empty content.",
        )

    content = content.strip()
    truncated = len(content) > MAX_CONTENT_CHARACTERS
    content = content[:MAX_CONTENT_CHARACTERS]

    return ExtractionResponse(
        url=extracted.get("url", url),
        status="EXTRACTED",
        content=content,
        character_count=len(content),
        truncated=truncated,
    )
