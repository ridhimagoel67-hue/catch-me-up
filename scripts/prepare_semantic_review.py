
import re
from pathlib import Path

import pandas as pd

from catchmeup.engine.chunking import chunk_all
from scripts.test_rag import load_units


SOURCE = Path("data/benchmark/answer_safety_results.csv")
DATA = Path("data/processed/gossip-girl_s1.json")
OUTPUT = Path("data/benchmark/answer_semantic_review.csv")

CITATION_PATTERN = re.compile(r"[\[【]([^\]】]+)[\]】]")


def main():
    if not SOURCE.is_file():
        raise FileNotFoundError(f"Safety report not found: {SOURCE}")

    if not DATA.is_file():
        raise FileNotFoundError(f"Content data not found: {DATA}")

    results = pd.read_csv(SOURCE, dtype={"id": str})

    units = load_units(DATA)
    chunks = chunk_all(units)
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}

    review_rows = []

    for _, row in results.iterrows():
        answer = str(row.get("answer", "") or "")
        cited_ids = CITATION_PATTERN.findall(answer)

        cited_evidence = []
        for chunk_id in cited_ids:
            chunk = chunks_by_id.get(chunk_id)

            if chunk is None:
                cited_evidence.append(
                    f"[{chunk_id}] NOT FOUND IN CURRENT DATASET"
                )
            else:
                cited_evidence.append(
                    f"[{chunk_id}] Position {chunk.position}\n"
                    f"{chunk.text}"
                )

        is_refusal = bool(row.get("refusal", False))

        review_rows.append({
            "id": row["id"],
            "question": row["question"],
            "answer": answer,
            "expected_behavior": row["expected_behavior"],
            "safety_assessment": row["safety_assessment"],
            "citation_valid": row["citation_valid"],
            "cited_chunk_ids": ";".join(cited_ids),
            "cited_evidence_text": "\n\n".join(cited_evidence),
            "answer_supports_question": (
                "NOT_APPLICABLE_REFUSAL"
                if is_refusal else "PENDING_MANUAL_REVIEW"
            ),
            "citations_support_claims": (
                "NOT_APPLICABLE_REFUSAL"
                if is_refusal else "PENDING_MANUAL_REVIEW"
            ),
            "review_notes": "",
        })

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(review_rows).to_csv(
        OUTPUT, index=False, encoding="utf-8-sig"
    )

    print(f"Prepared {len(review_rows)} review rows.")
    print(f"Saved to: {OUTPUT}")
    print("Original safety report was not modified.")


if __name__ == "__main__":
    main()
