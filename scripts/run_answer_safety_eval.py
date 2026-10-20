
import csv
import json
from pathlib import Path

import pandas as pd

from catchmeup.engine.citations import (
    REFUSAL_PHRASE,
    validate_citations,
)
from catchmeup.engine.chunking import chunk_all
from catchmeup.engine.embedding import embed
from catchmeup.engine.retriever import Retriever
from catchmeup.engine.spoiler_guard import should_protect_future
from scripts.test_rag import build_chain, load_units


BENCHMARK = Path("data/benchmark/questions_gg_s1.csv")
DATA = Path("data/processed/gossip-girl_s1.json")
REPORT = Path("data/benchmark/answer_safety_results.csv")
TOP_K = 10


def expected_behavior(row):
    """Determine the expected behavior from benchmark annotations."""
    question_type = str(row["type"]).strip().upper()
    asked_at = int(row["asked_at"])
    reveal_position = int(row["reveal_position"])

    if question_type == "A":
        return "ANSWER"

    if question_type == "B" and asked_at < reveal_position:
        return "PROTECT_FUTURE"

    if question_type == "B":
        return "ANSWER_IF_SUPPORTED"

    if question_type == "C":
        return "OUTSIDE_CURRENT_COLLECTION"

    if question_type == "D":
        return "UNRESOLVED_MYSTERY"

    return "REVIEW_LABEL"


def assess_safety(
    behavior,
    is_refusal,
    citation_valid,
    future_retrieval_leak,
    answer_present,
):
    """Assess observable safety, separately from semantic support."""

    if future_retrieval_leak:
        return "FAIL_FUTURE_RETRIEVAL_LEAK"

    if not answer_present:
        return "FAIL_EMPTY_RESPONSE"

    if not citation_valid:
        return "FAIL_INVALID_CITATIONS"

    if behavior in {
        "PROTECT_FUTURE",
        "OUTSIDE_CURRENT_COLLECTION",
        "UNRESOLVED_MYSTERY",
    }:
        if is_refusal:
            return "PASS_REFUSED"
        return "FAIL_UNSAFE_ANSWER"

    if is_refusal:
        return "REVIEW_UNEXPECTED_REFUSAL"

    return "REVIEW_SEMANTIC_SUPPORT"


def requires_manual_review(behavior, safety_assessment):
    """Flag protected cases and all review/failure outcomes."""

    requires_review_by_behavior = behavior in {
        "PROTECT_FUTURE",
        "OUTSIDE_CURRENT_COLLECTION",
        "UNRESOLVED_MYSTERY",
    }

    requires_review_by_assessment = str(safety_assessment).startswith(
        ("REVIEW_", "FAIL_")
    )

    return (
        requires_review_by_behavior
        or requires_review_by_assessment
    )


def empty_result(row, behavior, error):
    """Create a consistent report row when an evaluation errors."""

    return {
        "id": str(row["id"]),
        "type": str(row["type"]),
        "asked_at": int(row["asked_at"]),
        "reveal_position": int(row["reveal_position"]),
        "expected_behavior": behavior,
        "question": str(row["question"]),
        "answer": "",
        "guard_triggered": False,
        "refusal": False,
        "answer_present": False,
        "safety_assessment": "FAIL_EMPTY_RESPONSE",
        "retrieved_chunk_count": 0,
        "retrieved_chunk_ids": "",
        "future_retrieval_leak": False,
        "citation_valid": False,
        "citation_issues": "",
        "observed_result": "ERROR",
        "manual_safety_review_required": True,
        "response_metadata": "",
        "usage_metadata": "",
        "error": error,
    }


def main():
    # 1. Validate inputs before doing any evaluation work.
    if not BENCHMARK.is_file():
        raise FileNotFoundError(f"Benchmark not found: {BENCHMARK}")

    if not DATA.is_file():
        raise FileNotFoundError(f"Content data not found: {DATA}")

    questions = pd.read_csv(BENCHMARK, dtype={"id": str})

    required = {
        "id",
        "type",
        "asked_at",
        "reveal_position",
        "question",
        "gold_answer",
    }

    missing = required - set(questions.columns)

    if missing:
        raise ValueError(
            f"Benchmark missing columns: {sorted(missing)}"
        )

    if questions["id"].duplicated().any():
        duplicates = questions.loc[
            questions["id"].duplicated(), "id"
        ].tolist()
        raise ValueError(f"Duplicate question IDs: {duplicates}")

    # 2. Load content and construct the retriever.
    units = load_units(DATA)
    chunks = chunk_all(units)

    if not chunks:
        raise ValueError("No chunks were created from the input content.")

    retriever = Retriever(chunks, embed)

    # This initializes the model client; model calls happen in the loop.
    chain = build_chain()

    results = []

    print("=" * 72)
    print("ACTUAL ANSWER SAFETY EVALUATION")
    print("Model calls may be made for benchmark questions.")
    print("=" * 72)

    # 3. Evaluate each question at its annotated asking position.
    for _, row in questions.iterrows():
        question_id = str(row["id"])
        question = str(row["question"])
        position = int(row["asked_at"])
        reveal_position = int(row["reveal_position"])
        behavior = expected_behavior(row)

        print(f"\n[{question_id}] Position {position}: {question}")

        response_metadata = ""
        usage_metadata = ""

        try:
            # Retrieve only content available at the current position.
            hits = retriever.search(
                question,
                position,
                k=TOP_K,
            )

            future_chunks = [
                chunk.chunk_id
                for chunk, _ in hits
                if chunk.position > position
            ]

            # Fail closed if retrieval violates the position boundary.
            if future_chunks:
                raise AssertionError(
                    f"Future chunks retrieved: {future_chunks}"
                )

            evidence = "\n\n".join(
                f"Chunk ID: {chunk.chunk_id}\n"
                f"Position: {chunk.position}\n"
                f"Label: {chunk.label}\n"
                f"Text: {chunk.text}"
                for chunk, _ in hits
            )

            # Protect answers whose reveal position is still in the future.
            guard_triggered = (
                behavior == "PROTECT_FUTURE"
                and should_protect_future(position, reveal_position)
            )

            if guard_triggered:
                answer = REFUSAL_PHRASE
                citation_valid = True
                citation_issues = []

            elif not hits:
                answer = REFUSAL_PHRASE
                citation_valid = True
                citation_issues = []

            else:
                # Supply every variable expected by the answer chain.
                response = chain.invoke(
                    {
                        "collection": "Gossip Girl Season 1",
                        "unit_name": "episode",
                        "position": position,
                        "question": question,
                        "evidence": evidence,
                        "retry_feedback": "",
                    }
                )

                raw_content = response.content
                answer = (
                    raw_content
                    if isinstance(raw_content, str)
                    else ""
                ).strip()

                response_metadata = json.dumps(
                    response.response_metadata or {},
                    default=str,
                )

                usage_metadata = json.dumps(
                    response.usage_metadata or {},
                    default=str,
                )

                is_model_refusal = answer.startswith(REFUSAL_PHRASE)

                citation_valid, citation_issues = validate_citations(
                    answer,
                    {chunk.chunk_id for chunk, _ in hits},
                    allow_uncited=is_model_refusal,
                )

            # 4. Assess the final response consistently for every path.
            is_refusal = answer.startswith(REFUSAL_PHRASE)
            answer_present = bool(answer)

            if not answer_present:
                observed_result = "EMPTY_RESPONSE"
            elif not citation_valid:
                observed_result = "INVALID_CITATIONS"
            elif is_refusal:
                observed_result = "REFUSED"
            else:
                observed_result = "ANSWERED"

            safety_assessment = assess_safety(
                behavior=behavior,
                is_refusal=is_refusal,
                citation_valid=citation_valid,
                future_retrieval_leak=bool(future_chunks),
                answer_present=answer_present,
            )

            # 5. Build one row for the report.
            result = {
                "id": question_id,
                "type": str(row["type"]),
                "asked_at": position,
                "reveal_position": reveal_position,
                "expected_behavior": behavior,
                "question": question,
                "answer": answer,
                "guard_triggered": guard_triggered,
                "refusal": is_refusal,
                "answer_present": answer_present,
                "safety_assessment": safety_assessment,
                "retrieved_chunk_count": len(hits),
                "retrieved_chunk_ids": ";".join(
                    chunk.chunk_id for chunk, _ in hits
                ),
                "future_retrieval_leak": bool(future_chunks),
                "citation_valid": citation_valid,
                "citation_issues": "; ".join(citation_issues),
                "observed_result": observed_result,
                "manual_safety_review_required": requires_manual_review(
                    behavior,
                    safety_assessment,
                ),
                "response_metadata": response_metadata,
                "usage_metadata": usage_metadata,
                "error": "",
            }

        except Exception as error:
            result = empty_result(
                row,
                behavior,
                f"{type(error).__name__}: {error}",
            )
            result["response_metadata"] = response_metadata
            result["usage_metadata"] = usage_metadata

        results.append(result)

        print("Expected:", result["expected_behavior"])
        print("Observed:", result["observed_result"])
        print("Safety assessment:", result["safety_assessment"])
        print(
            "Manual review required:",
            result["manual_safety_review_required"],
        )

        if result["error"]:
            print("Error:", result["error"])
        else:
            print("Answer:", result["answer"][:500])

    if not results:
        raise RuntimeError("No benchmark questions were evaluated.")

    # 6. Write a consistent CSV report.
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    with REPORT.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(results[0].keys()),
        )
        writer.writeheader()
        writer.writerows(results)

    # 7. Summarize outcomes and safety checks.
    print("\n" + "=" * 72)
    print("EVALUATION SUMMARY")
    print("=" * 72)

    total = len(results)

    print("Questions:", total)
    print(
        "Answers present:",
        sum(result["answer_present"] for result in results),
        "/",
        total,
    )
    print("Refusals:", sum(result["refusal"] for result in results))
    print(
        "Valid citation checks:",
        sum(result["citation_valid"] for result in results),
        "/",
        total,
    )
    print(
        "Future retrieval leaks:",
        sum(result["future_retrieval_leak"] for result in results),
    )
    print("Errors:", sum(bool(result["error"]) for result in results))
    print(
        "Manual reviews required:",
        sum(
            result["manual_safety_review_required"]
            for result in results
        ),
    )
    print("Results saved to:", REPORT)

    print(
        "\nImportant: citation-ID validity does not prove semantic "
        "support. Answer quality and benchmark annotations still "
        "require human review."
    )

    print("\nSafety assessment counts:")
    for status in sorted(
        {result["safety_assessment"] for result in results}
    ):
        count = sum(
            result["safety_assessment"] == status
            for result in results
        )
        print(f"{status}: {count}")

    print(
        "Deterministic guard triggers:",
        sum(result["guard_triggered"] for result in results),
    )


if __name__ == "__main__":
    main()
