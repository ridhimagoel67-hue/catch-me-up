
import json
from pathlib import Path

import pandas as pd

from catchmeup.engine.chunking import chunk_all


DATA = Path("data/processed/gossip-girl_s1.json")
QUESTIONS = Path("data/benchmark/questions_gg_s1.csv")

QUESTION_ID = "q04"
MAX_RESULTS = 20

KEYWORDS = {
    "lily",
    "dan",
    "serena",
    "approve",
    "approval",
    "relationship",
    "dating",
    "mother",
    "together",
}


def main():
    # 1. Load the episode data and benchmark.
    units = json.loads(DATA.read_text(encoding="utf-8"))
    questions = pd.read_csv(QUESTIONS)

    matching_questions = questions[
        questions["id"] == QUESTION_ID
    ]

    if matching_questions.empty:
        raise ValueError(
            f"Question {QUESTION_ID} was not found in the benchmark."
        )

    question_row = matching_questions.iloc[0]

    question = question_row["question"]
    asked_at = int(question_row["asked_at"])

    # 2. Create chunks using the project's existing chunker.
    chunks = chunk_all(units)

    # 3. Enforce the viewer's position restriction.
    visible_chunks = [
        chunk for chunk in chunks
        if chunk.position <= asked_at
    ]

    # 4. Find chunks that mention relevant keywords.
    candidates = []

    for chunk in visible_chunks:
        text = chunk.text.casefold()

        matched_keywords = sorted(
            keyword
            for keyword in KEYWORDS
            if keyword in text
        )

        # Require at least two matching keywords.
        if len(matched_keywords) >= 2:
            candidates.append(
                (chunk, matched_keywords)
            )

    # 5. Rank candidates by keyword overlap.
    candidates.sort(
        key=lambda item: (
            -len(item[1]),
            item[0].position,
            item[0].chunk_id,
        )
    )

    # 6. Print evidence for manual inspection.
    print(f"Question: {question}")
    print(f"Viewer position: {asked_at}")
    print(f"Visible chunks: {len(visible_chunks)}")
    print(f"Matching candidate chunks: {len(candidates)}")

    print("\nCANDIDATE EVIDENCE")
    print("=" * 70)

    for chunk, matched_keywords in candidates[:MAX_RESULTS]:
        print(
            f"\n{chunk.chunk_id} | "
            f"position={chunk.position} | "
            f"keywords={', '.join(matched_keywords)}"
        )
        print(chunk.text)

    print("\n" + "=" * 70)
    print("Inspection complete.")
    print("Keyword overlap is only a heuristic, not proof of relevance.")


if __name__ == "__main__":
    main()