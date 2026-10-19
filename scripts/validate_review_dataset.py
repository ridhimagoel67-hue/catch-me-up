
import csv
import json
from collections import Counter
from pathlib import Path

from catchmeup.engine.chunking import chunk_all


SOURCE_FILE = Path("data/processed/gossip-girl_s1.json")
GENERATED_FILE = Path("data/processed/generated_questions.jsonl")
REVIEW_FILE = Path("data/benchmark/generated_questions_review_corrected.csv")
REPORT_FILE = Path("data/benchmark/review_validation_report.csv")


def load_jsonl(path):
    """Load JSONL records and report malformed lines."""

    records = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                print(f"Malformed JSONL line {line_number}: {error}")

    return records


def load_chunks():
    """Recreate the original chunks from the episode data."""

    with SOURCE_FILE.open("r", encoding="utf-8") as file:
        units = json.load(file)

    chunks = chunk_all(units)

    # Each chunk ID should map to exactly one source chunk.
    return {chunk.chunk_id: chunk for chunk in chunks}


def validate():
    for path in (SOURCE_FILE, GENERATED_FILE, REVIEW_FILE):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    generated_records = load_jsonl(GENERATED_FILE)
    chunks = load_chunks()

    # Preserve all records per ID so duplicate IDs can be detected.
    generated_by_id = {}
    for record in generated_records:
        chunk_id = record.get("chunk_id")
        if chunk_id:
            generated_by_id.setdefault(chunk_id, []).append(record)

    with REVIEW_FILE.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        review_rows = list(csv.DictReader(file))

    review_ids = [row.get("id", "").strip() for row in review_rows]
    review_id_counts = Counter(review_ids)

    report_rows = []

    for row in review_rows:
        review_id = row.get("id", "").strip()

        # Exported IDs have the form gen-s01e01-c01.
        chunk_id = (
            review_id.removeprefix("gen-")
            if review_id.startswith("gen-")
            else ""
        )

        issues = []

        if not review_id:
            issues.append("missing_review_id")

        if review_id and review_id_counts[review_id] > 1:
            issues.append("duplicate_review_id")

        if not chunk_id or chunk_id not in chunks:
            issues.append("source_chunk_not_found")

        matching_records = generated_by_id.get(chunk_id, [])

        if not matching_records:
            issues.append("generated_record_not_found")
        elif len(matching_records) > 1:
            issues.append("duplicate_generated_records")

        source_chunk = chunks.get(chunk_id)
        generated_record = (
            matching_records[0] if len(matching_records) == 1 else None
        )

        if source_chunk:
            if row.get("evidence_unit", "").strip() != source_chunk.unit_id:
                issues.append("evidence_unit_mismatch")

            try:
                review_position = int(row.get("reveal_position", ""))
                if review_position != source_chunk.position:
                    issues.append("reveal_position_mismatch")
            except (TypeError, ValueError):
                issues.append("invalid_reveal_position")

        if generated_record:
            original_quote = generated_record.get(
                "support_quote",
                generated_record.get("evidence", ""),
            )
            review_quote = row.get("evidence", "")

            if not isinstance(original_quote, str) or not original_quote:
                issues.append("original_quote_missing")
            elif review_quote != original_quote:
                issues.append("evidence_differs_from_original_quote")

            if source_chunk and isinstance(original_quote, str):
                if original_quote and original_quote not in source_chunk.text:
                    issues.append("original_quote_not_in_source_chunk")

            original_question = generated_record.get("question", "")
            original_answer = generated_record.get(
                "answer",
                generated_record.get("gold_answer", ""),
            )

            if row.get("question", "").strip() != str(original_question).strip():
                issues.append("question_changed_from_original")

            if row.get("gold_answer", "").strip() != str(original_answer).strip():
                issues.append("answer_changed_from_original")

        for field in ("question", "gold_answer", "evidence"):
            if not row.get(field, "").strip():
                issues.append(f"missing_{field}")

        report_rows.append({
            "id": review_id,
            "chunk_id": chunk_id,
            "review_status": row.get("review_status", ""),
            "validation_status": "PASS" if not issues else "CHECK",
            "issues": "; ".join(issues),
        })

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with REPORT_FILE.open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "id",
                "chunk_id",
                "review_status",
                "validation_status",
                "issues",
            ],
        )
        writer.writeheader()
        writer.writerows(report_rows)

    status_counts = Counter(
        row["validation_status"] for row in report_rows
    )
    issue_counts = Counter(
        issue
        for row in report_rows
        for issue in row["issues"].split("; ")
        if issue
    )

    print("\nValidation finished.")
    print("Review rows:", len(review_rows))
    print("Generated records:", len(generated_records))
    print("Source chunks:", len(chunks))
    print("Rows passing checks:", status_counts["PASS"])
    print("Rows needing checks:", status_counts["CHECK"])

    if issue_counts:
        print("\nIssues found:")
        for issue, count in sorted(issue_counts.items()):
            print(f"  {issue}: {count}")
    else:
        print("No structural or evidence-matching issues found.")

    print("\nReport saved to:", REPORT_FILE)
    print("No source or review files were modified.")


if __name__ == "__main__":
    validate()
