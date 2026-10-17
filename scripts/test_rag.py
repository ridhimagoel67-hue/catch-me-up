
import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from catchmeup.engine.chunking import chunk_all
from catchmeup.engine.embedding import embed
from catchmeup.engine.retriever import Retriever
from catchmeup.engine.citations import (
    REFUSAL_PHRASE,
    validate_citations,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ask evidence-grounded questions about ordered content."
    )
    parser.add_argument(
        "--data",
        default="data/processed/gossip-girl_s1.json",
        help="Path to a JSON file containing ordered content units.",
    )
    parser.add_argument(
        "--collection",
        default="this collection",
        help="Name of the collection, e.g. a show, book, or course.",
    )
    parser.add_argument(
        "--unit-name",
        default="content unit",
        help="Name of a unit, e.g. episode, chapter, or lesson.",
    )
    return parser.parse_args()


def load_units(data_path):
    path = Path(data_path)

    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        units = json.load(file)

    if not isinstance(units, list) or not units:
        raise ValueError("The dataset must be a non-empty JSON list.")

    required_fields = {"unit_id", "position", "label", "text"}

    for index, unit in enumerate(units):
        if not isinstance(unit, dict):
            raise ValueError(f"Unit {index} must be a JSON object.")

        missing = required_fields - unit.keys()
        if missing:
            raise ValueError(
                f"Unit {index} is missing fields: {sorted(missing)}"
            )

        if not isinstance(unit["position"], int):
            raise ValueError(
                f"Unit {index} must have an integer position."
            )

    return units


def build_chain():
    load_dotenv(encoding="utf-8-sig")

    required_env = ["LLM_MODEL", "LLM_API_KEY", "LLM_BASE_URL"]
    missing_env = [
        key for key in required_env if not os.environ.get(key)
    ]

    if missing_env:
        raise RuntimeError(
            "Missing required environment variables: "
            + ", ".join(missing_env)
        )

    llm = ChatOpenAI(
        model=os.environ["LLM_MODEL"],
        api_key=os.environ["LLM_API_KEY"],
        base_url=os.environ["LLM_BASE_URL"],
        max_tokens=300,
        temperature=0,
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are Catch Me Up, an evidence-grounded question-answering
assistant for user-provided content.

CORE RULES
1. Answer using ONLY the supplied evidence. Do not use outside knowledge,
   memory, assumptions, or facts from later content.
2. Answer the exact question asked. Do not replace a question about
   someone's identity with a description of their role or actions.
3. For questions asking WHO a person is, provide the person's identity
   only if the supplied evidence establishes it. Describing someone as
   a narrator, account, author, or character does not establish the
   real or fictional person's identity.
4. For questions asking whether something eventually happens, do not
   reveal outcomes that are not established by evidence available at
   the user's current position.
5. If the evidence does not adequately answer the question, say exactly:
   "I couldn't find enough evidence in the content available to you."
6. Do not guess, infer an unsupported identity, or present speculation
   as fact.
7. Cite supporting chunks using their exact IDs, for example
   [unit-01-c01]. Do not invent citation IDs.
8. If you cannot support the answer with the supplied evidence, prefer
   the refusal sentence over a plausible but unsupported answer.

Keep the answer concise."""
        ),
        (
            "human",
            """Collection: {collection}
Content type: {unit_name}
User's current position: {position}

Question: {question}

Available evidence:
{evidence}

Answer the exact question using only this evidence.
Cite the chunks that support your answer."""
        ),
    ])

    return prompt | llm


def main():
    args = parse_args()
    units = load_units(args.data)
    chunks = chunk_all(units)

    print(
        f"Loaded {len(units)} units and {len(chunks)} chunks "
        f"from {args.collection}."
    )

    retriever = Retriever(chunks, embed)
    chain = build_chain()

    min_position = min(unit["position"] for unit in units)
    max_position = max(unit["position"] for unit in units)

    question = input("Your question: ").strip()

    if not question:
        print("Please enter a question.")
        return

    try:
        position = int(
            input(f"Your current {args.unit_name} position: ").strip()
        )
    except ValueError:
        print("Please enter a valid integer position.")
        return

    if not min_position <= position <= max_position:
        print(
            f"Please enter a position between {min_position} "
            f"and {max_position}."
        )
        return

    hits = retriever.search(question, position, k=10)

    if not hits:
        print(f"\n{REFUSAL_PHRASE}")
        print("Citation check: PASSED (no answer generated)")
        return

    assert all(
        chunk.position <= position for chunk, _ in hits
    ), "A future-position chunk was retrieved."

    evidence = "\n\n".join(
        f"Chunk ID: {chunk.chunk_id}\n"
        f"Position: {chunk.position}\n"
        f"Label: {chunk.label}\n"
        f"Text: {chunk.text}"
        for chunk, _ in hits
    )

    print("\nRetrieved evidence:")

    for chunk, score in hits:
        print(
            f"\n{chunk.chunk_id} | position {chunk.position} "
            f"| score {score:.3f}"
        )
        print(chunk.text[:250])

    response = chain.invoke({
        "collection": args.collection,
        "unit_name": args.unit_name,
        "position": position,
        "question": question,
        "evidence": evidence,
    })

    answer = response.content

    print("\n" + "=" * 50)
    print("GENERATED ANSWER")
    print("=" * 50)

    if not isinstance(answer, str) or not answer.strip():
        print("[EMPTY MODEL RESPONSE]")
        print("=" * 50)
        print("\nThe model returned no answer text.")
        print("Response metadata:", response.response_metadata)
        print("Usage metadata:", response.usage_metadata)
        return

    print(answer)
    print("=" * 50)

    supplied_chunk_ids = {
        chunk.chunk_id for chunk, _ in hits
    }

    is_refusal = answer.strip().startswith(REFUSAL_PHRASE)

    citations_valid, citation_issues = validate_citations(
        answer,
        supplied_chunk_ids,
        allow_uncited=is_refusal,
    )

    if citations_valid:
        print("\nCitation check: PASSED")
    else:
        print("\nCitation check: FAILED")
        print("Citation issues:", citation_issues)

    


if __name__ == "__main__":
    main()

