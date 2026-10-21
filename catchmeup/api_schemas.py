
from pydantic import BaseModel, Field, ConfigDict


class AskRequest(BaseModel):
    """Public input accepted by the Catch Me Up API."""

    model_config = ConfigDict(extra="forbid")

    collection_id: str = Field(
        default="gossip-girl-s1",
        min_length=1,
        max_length=100,
        description="ID of a collection registered by the server.",
    )

    question: str = Field(
        min_length=1,
        max_length=2000,
        description="Question about the content available to the user.",
    )

    current_position: int = Field(
        ge=0,
        description="The user's current episode, chapter, or lesson position.",
    )


class AskResponse(BaseModel):
    """Structured answer returned by the API."""

    answer: str
    status: str
    citation_valid: bool | None = None
    citation_issues: list[str] = Field(default_factory=list)
    refusal_reason: str | None = None
    spoiler_protected: bool = False
    safety_flags: list[str] = Field(default_factory=list)
    retry_count: int = 0

