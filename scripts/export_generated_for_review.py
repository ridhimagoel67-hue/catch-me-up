
import csv
import json
from pathlib import Path


INPUT_FILE = Path("data/processed/generated_questions.jsonl")
OUTPUT_FILE = Path("data/benchmark/generated_questions_review.csv")

OUTPUT_COLUMNS = [
    "id",
    "type",
    "asked_at",
    "reveal_position",
    "evidence_unit",
    "question",
    "gold_answer",
    "evidence",
    "review_status",
]


def load_records():
    """Load generated questions from the JSONL file."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    records = []

    with INPUT_FILE.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                print(
                    f"Skipping malformed JSON on line "
                    f"{line_number}: {error}"
                )
                continue

            records.append(record)

    return records


def prepare_review_records(records):
    """Convert generated records into the review CSV schema."""

    review_records = []
    seen_chunks = set()

    for record in records:
        chunk_id = record.get("chunk_id")

        if not chunk_id:
            print("Skipping record without chunk_id.")
            continue

        # Avoid exporting the same chunk more than once.
        if chunk_id in seen_chunks:
            print(f"Skipping duplicate chunk: {chunk_id}")
            continue

        seen_chunks.add(chunk_id)

        # Explicitly rejected chunks are not question candidates.
        if record.get("suitable") is False:
            continue

        question = record.get("question", "")
        answer = record.get("answer", record.get("gold_answer", ""))
        evidence = record.get(
            "support_quote",
            record.get("evidence", ""),
        )

        # Skip legacy or generated records missing essential fields.
        if not all(
            isinstance(value, str) and value.strip()
            for value in (question, answer, evidence)
        ):
            print(f"Skipping incomplete record: {chunk_id}")
            continue

        review_records.append({
            "id": f"gen-{chunk_id}",
            "type": "",
            "asked_at": "",
            "reveal_position": record.get("reveal_position", ""),
            "evidence_unit": record.get("unit_id", ""),
            "question": question.strip(),
            "gold_answer": answer.strip(),
            "evidence": evidence.strip(),
            "review_status": "pending",
        })

    return review_records


def main():
    records = load_records()
    review_records = prepare_review_records(records)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=OUTPUT_COLUMNS,
        )
        writer.writeheader()
        writer.writerows(review_records)

    print("\nExport complete!")
    print("Input records:", len(records))
    print("Review candidates:", len(review_records))
    print("Output file:", OUTPUT_FILE)
    print("Original benchmark was not modified.")


if __name__ == "__main__":
    main()
