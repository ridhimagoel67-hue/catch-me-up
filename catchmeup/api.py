from pydantic import BaseModel, Field
from catchmeup.ingestion import ingest_episode_collection
from catchmeup.extraction import (
    ExtractionRequest,
    ExtractionResponse,
    extract_source,
)
from catchmeup.discovery import (
    DiscoveryRequest,
    DiscoveryResponse,
    discover_sources,
)
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse

from catchmeup.api_schemas import AskRequest, AskResponse
from catchmeup.collections import get_collection
from catchmeup.engine.chunking import chunk_all
from catchmeup.engine.embedding import embed
from catchmeup.engine.retriever import Retriever
from catchmeup.engine.workflow import build_workflow
from scripts.test_rag import build_chain, load_units


DEFAULT_COLLECTION_ID = "gossip-girl-s1"
MAX_RETRIES = 2

class IngestRequest(BaseModel):
    collection_id: str = Field(min_length=1, max_length=100)
    name: str = Field(min_length=1, max_length=200)
    source_url: str = Field(min_length=1, max_length=2000)
    content: str = Field(min_length=1, max_length=50000)


def build_runtime(config) -> dict[str, Any]:
    """Load one registered collection and build its answering dependencies."""

    units = load_units(config.data_path)

    if not units:
        raise ValueError(
            f"Collection '{config.collection_id}' contains no content units."
        )

    chunks = chunk_all(units)
    retriever = Retriever(chunks, embed)
    answer_chain = build_chain()

    positions = [unit["position"] for unit in units]

    return {
        "units": units,
        "retriever": retriever,
        "answer_chain": answer_chain,
        "min_position": min(positions),
        "max_position": max(positions),
    }


def create_app(
    retriever=None,
    answer_chain=None,
    units=None,
) -> FastAPI:
    """
    Create the API.

    Optional injected dependencies let tests run without loading the real
    embedding model, dataset, or LLM.
    """

    injected = (retriever, answer_chain, units)

    if any(item is not None for item in injected) and not all(
        item is not None for item in injected
    ):
        raise ValueError(
            "Provide retriever, answer_chain, and units together, "
            "or provide none of them."
        )

    runtime_lock = Lock()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.runtimes = {}

        if all(item is not None for item in injected):
            if not units:
                raise ValueError("Injected units cannot be empty.")

            positions = [unit["position"] for unit in units]

            default_runtime = {
                "units": units,
                "retriever": retriever,
                "answer_chain": answer_chain,
                "min_position": min(positions),
                "max_position": max(positions),
            }
        else:
            default_config = get_collection(DEFAULT_COLLECTION_ID)
            default_runtime = build_runtime(default_config)

        app.state.runtimes[DEFAULT_COLLECTION_ID] = default_runtime

        # Keep the existing default runtime available for compatibility.
        app.state.runtime = default_runtime
        app.state.workflow = build_workflow()

        yield

    app = FastAPI(
        title="Catch Me Up API",
        description=(
            "Evidence-grounded question answering over ordered content, "
            "with position-aware retrieval and citation validation."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse(url="/docs")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/ask", response_model=AskResponse)
    def ask(request: AskRequest):
        runtimes = getattr(app.state, "runtimes", None)
        workflow = getattr(app.state, "workflow", None)

        if runtimes is None or workflow is None:
            raise HTTPException(
                status_code=503,
                detail="The application has not finished initializing.",
            )

        # Resolve the requested collection using server-side configuration.
        try:
            config = get_collection(request.collection_id)
        except KeyError:
            raise HTTPException(
                status_code=404,
                detail="The requested collection is not registered.",
            ) from None
        except FileNotFoundError:
            raise HTTPException(
                status_code=503,
                detail="The requested collection's dataset is unavailable.",
            ) from None

        # Build each collection runtime only when first requested, then reuse it.
        if request.collection_id not in runtimes:
            with runtime_lock:
                if request.collection_id not in runtimes:
                    try:
                        runtimes[request.collection_id] = build_runtime(config)
                    except FileNotFoundError:
                        raise HTTPException(
                            status_code=503,
                            detail="The requested collection's dataset is unavailable.",
                        ) from None
                    except Exception:
                        raise HTTPException(
                            status_code=503,
                            detail="The requested collection could not be initialized.",
                        ) from None

        runtime = runtimes[request.collection_id]
        position = request.current_position

        if not (
            runtime["min_position"]
            <= position
            <= runtime["max_position"]
        ):
            raise HTTPException(
                status_code=422,
                detail=(
                    "current_position must be between "
                    f"{runtime['min_position']} and "
                    f"{runtime['max_position']}."
                ),
            )

        try:
            result = workflow.invoke(
                {
                    "question": request.question,
                    "current_position": position,
                    "collection_name": config.name,
                    "unit_name": config.unit_name,
                    "retriever": runtime["retriever"],
                    "answer_chain": runtime["answer_chain"],
                    "retry_count": 0,
                    "max_retries": MAX_RETRIES,
                    "retry_feedback": "",
                }
            )
        except Exception:
            # Never expose internal exception details to API clients.
            raise HTTPException(
                status_code=500,
                detail="The request could not be processed.",
            ) from None

        status = result.get("status", "ERROR")

        if status == "ERROR":
            raise HTTPException(
                status_code=503,
                detail="The answering workflow could not complete the request.",
            )

        return AskResponse(
            answer=result.get(
                "answer",
                "I couldn't find enough evidence in the content available to you.",
            ),
            status=status,
            citation_valid=result.get("citation_valid"),
            citation_issues=result.get("citation_issues", []),
            refusal_reason=result.get("refusal_reason") or None,
            spoiler_protected=result.get("spoiler_protected", False),
            safety_flags=result.get("safety_flags", []),
            retry_count=result.get("retry_count", 0),
        )

    @app.post(
        "/discover",
        response_model=DiscoveryResponse,
)
    def discover(request: DiscoveryRequest):
        return discover_sources(request)


    @app.post(
        "/extract",
        response_model=ExtractionResponse,
)
    def extract(request: ExtractionRequest):
        return extract_source(request)
    


    @app.post("/ingest")
    def ingest(request: IngestRequest):
        try:
            result = ingest_episode_collection(
                content=request.content,
                collection_id=request.collection_id,
                name=request.name,
                source_url=request.source_url,
            )
            return {
                "status": "REGISTERED",
                **result,
            }
        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail=str(exc),
            ) from None

    return app

     

app = create_app()
