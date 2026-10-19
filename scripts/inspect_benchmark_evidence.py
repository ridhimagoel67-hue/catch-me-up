
import json
from pathlib import Path

import pandas as pd

from catchmeup.engine.chunking import chunk_all


DATA = Path("data/processed/gossip-girl_s1.json")
QUESTIONS = Path("data/benchmark/questions_gg_s1.csv")


def main():
    units = json.loads(DATA.read_text(encoding="utf-8"))
    chunks = chunk_all(units)
    questions = pd.read_csv(QUESTIONS, dtype={"id": str})

    for _, row in questions.iterrows():
        question_id = row["id"]
        question = str(row["question"])
        asked_at = int(row["asked_at"])
        evidence_unit = row["evidence_unit"]

        # Only inspect content available at the asking position.
        available = [
            chunk for chunk in chunks
            if chunk.position <= asked_at
        ]

        print("\n" + "=" * 75)
        print(f"{question_id}: {question}")
        print(f"Asking position: {asked_at}")
        print(f"Evidence unit in benchmark: {evidence_unit}")
        print(f"Available chunks: {len(available)}")

        # Print the most relevant-looking chunks from the benchmark's
        # referenced evidence unit, if that unit is available yet.
        matching_unit = [
            chunk for chunk in available
            if chunk.unit_id == str(evidence_unit)
        ]

        if matching_unit:
            for chunk in matching_unit:
                print(f"\n{chunk.chunk_id} | {chunk.label}")
                print(chunk.text)
        else:
            print("Referenced evidence unit is not available at this position.")

    print("\nInspection complete. No files were modified.")


if __name__ == "__main__":
    main()
