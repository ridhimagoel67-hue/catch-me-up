
import csv
import json
from pathlib import Path
from catchmeup.engine.spoiler_guard import should_protect_future

import pandas as pd

from catchmeup.engine.chunking import chunk_all
from catchmeup.engine.embedding import embed
from catchmeup.engine.retriever import Retriever
from catchmeup.engine.citations import (
    REFUSAL_PHRASE,
    validate_citations,
)
from scripts.test_rag import build_chain, load_units


BENCHMARK = Path("data/benchmark/questions_gg_s1.csv")
DATA = Path("data/processed/gossip-girl_s1.json")
REPORT = Path("data/benchmark/answer_safety_results.csv")
TOP_K = 10


def expected_behavior(row):
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
    """Assess observed safety separately from citation-ID validity."""

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


def empty_result(row, behavior, error):
    """Return a result with the same fields as a successful evaluation."""
    return {
        "id": str(row["id"]),
        "type": str(row["type"]),
        "asked_at": int(row["asked_at"]),
        "reveal_position": int(row["reveal_position"]),
        "expected_behavior": behavior,
        "question": str(row["question"]),
        "guard_triggered": False,
        "answer": "",
        "refusal": False,
        "answer_present": False,
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
    if not BENCHMARK.is_file():
        raise FileNotFoundError(f"Benchmark not found: {BENCHMARK}")

    units = load_units(DATA)
    chunks = chunk_all(units)

    retriever = Retriever(chunks, embed)
    chain = build_chain()

    questions = pd.read_csv(BENCHMARK)

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

    results = []

    print("=" * 72)
    print("ACTUAL ANSWER SAFETY EVALUATION")
    print("Model calls will be made for each benchmark question.")
    print("=" * 72)

    for _, row in questions.iterrows():
        question_id = str(row["id"])
        question = str(row["question"])
        position = int(row["asked_at"])
        behavior = expected_behavior(row)

        print(f"\n[{question_id}] Position {position}: {question}")

        response_metadata = ""
        usage_metadata = ""

        try:
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

            # Never send future-position evidence to the model.
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

            
            guard_triggered = (
            behavior == "PROTECT_FUTURE"
            and should_protect_future(
                position,
                int(row["reveal_position"]),
            )
        )

            if guard_triggered:
                # The annotated outcome has not been revealed.
                # Do not send this question to the LLM.
                answer = REFUSAL_PHRASE
                citation_valid = True
                citation_issues = []

            elif not hits:
                answer = REFUSAL_PHRASE
                citation_valid = True
                citation_issues = []

            else:
                response = chain.invoke({
                    "collection": "Gossip Girl Season 1",
                    "unit_name": "episode",
                    "position": position,
                    "question": question,
                    "evidence": evidence,
                })

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

                if not answer:
                    print("WARNING: Empty model response.")
                    print("Response metadata:", response_metadata)
                    print("Usage metadata:", usage_metadata)

                is_refusal = answer.startswith(REFUSAL_PHRASE)

                citation_valid, citation_issues = validate_citations(
                    answer,
                    {chunk.chunk_id for chunk, _ in hits},
                    allow_uncited=is_refusal,
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

                if not answer:
                    print("WARNING: Empty model response.")
                    print("Response metadata:", response_metadata)
                    print("Usage metadata:", usage_metadata)

                is_refusal = answer.startswith(REFUSAL_PHRASE)

                citation_valid, citation_issues = validate_citations(
                    answer,
                    {chunk.chunk_id for chunk, _ in hits},
                    allow_uncited=is_refusal,
                )

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

            result = {
                "id": question_id,
                "type": str(row["type"]),
                "asked_at": position,
                "reveal_position": int(row["reveal_position"]),
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
                "manual_safety_review_required": behavior in {
                    "PROTECT_FUTURE",
                    "OUTSIDE_CURRENT_COLLECTION",
                    "UNRESOLVED_MYSTERY",
                },
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
        print("Citations valid:", result["citation_valid"])

        if result["error"]:
            print("Error:", result["error"])
        else:
            print("Answer:", result["answer"][:500])

    if not results:
        raise RuntimeError("No benchmark questions were evaluated.")

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
    print(
        "Refusals:",
        sum(result["refusal"] for result in results),
    )
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
    print(
        "Errors:",
        sum(bool(result["error"]) for result in results),
    )

    print("Results saved to:", REPORT)
    print(
        "\nImportant: citation-ID validity does not prove semantic "
        "support. Future-answer safety still requires review."
    )
    print("\nSafety assessment counts:")

    for status in sorted({
        result["safety_assessment"] for result in results
    }):
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
