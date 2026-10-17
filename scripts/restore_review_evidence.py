
import argparse
import csv
import json
from pathlib import Path

from catchmeup.engine.chunking import chunk_all


ROOT = Path(__file__).resolve().parents[1]

GENERATED_FILE = ROOT / "data" / "processed" / "generated_questions.jsonl"
SOURCE_FILE = ROOT / "data" / "processed" / "gossip-girl_s1.json"
OUTPUT_FILE = ROOT / "data" / "benchmark" / "generated_questions_review_corrected.csv"
REPORT_FILE = ROOT / "data" / "benchmark" / "evidence_restore_report.csv"


def load_jsonl(path):
    """Read one JSON object per line."""
    records = []

    with path.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path}, line {line_number}: {exc}"
                ) from exc

    return records


def load_source_chunks():
    """Re-create chunks from the original episode data."""
    with SOURCE_FILE.open("r", encoding="utf-8-sig") as file:
        units = json.load(file)

    chunks = chunk_all(units)

    return {
        chunk.chunk_id: chunk
        for chunk in chunks
    }


def main():
    parser = argparse.ArgumentParser(
        description="Restore original evidence quotes in a reviewed CSV."
    )
    parser.add_argument(
        "review_csv",
        help="Path to your renamed final review CSV, relative to the project "
             "root or as an absolute path.",
    )
    args = parser.parse_args()

    review_path = Path(args.review_csv)
    if not review_path.is_absolute():
        review_path = ROOT / review_path

    if not review_path.exists():
        raise FileNotFoundError(
            f"Review CSV not found: {review_path}\n"
            "Check the filename and path, then run the script again."
        )

    for required_file in (GENERATED_FILE, SOURCE_FILE):
        if not required_file.exists():
            raise FileNotFoundError(
                f"Required source file not found: {required_file}"
            )

    # Load the original generated questions.
    generated_records = load_jsonl(GENERATED_FILE)

    generated_by_chunk = {}
    for record in generated_records:
        chunk_id = record.get("chunk_id")

        if not chunk_id:
            continue

        if chunk_id in generated_by_chunk:
            raise ValueError(
                f"Duplicate chunk_id in generated JSONL: {chunk_id}"
            )

        generated_by_chunk[chunk_id] = record

    # Re-create the source chunks so we can verify each original quote.
    source_chunks = load_source_chunks()

    # Read the user's reviewed CSV without changing the original.
    with review_path.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames

        if not fieldnames:
            raise ValueError("The review CSV has no header row.")

        rows = list(reader)

    if "id" not in fieldnames or "evidence" not in fieldnames:
        raise ValueError(
            "The CSV must contain at least 'id' and 'evidence' columns."
        )

    output_rows = []
    report_rows = []
    seen_ids = set()

    for row_number, row in enumerate(rows, start=2):
        row = dict(row)
        question_id = (row.get("id") or "").strip()

        issues = []
        chunk_id = question_id.removeprefix("gen-")

        if not question_id:
            issues.append("missing_id")

        if question_id in seen_ids:
            issues.append("duplicate_review_id")
        seen_ids.add(question_id)

        original = generated_by_chunk.get(chunk_id)
        chunk = source_chunks.get(chunk_id)

        if original is None:
            issues.append("original_generated_record_not_found")

        if chunk is None:
            issues.append("source_chunk_not_found")

        original_quote = ""
        quote_status = "NOT_RESTORED"

        if original is not None:
            # Newer records use support_quote; older ones may use evidence.
            original_quote = (
                original.get("support_quote")
                or original.get("evidence")
                or ""
            ).strip()

        if not original_quote:
            issues.append("original_quote_missing")
            quote_status = "MISSING_ORIGINAL_QUOTE"

        elif chunk is None:
            quote_status = "SOURCE_CHUNK_MISSING"

        elif original_quote not in chunk.text:
            issues.append("original_quote_not_found_in_source_chunk")
            quote_status = "QUOTE_NOT_VERIFIED"

        else:
            current_evidence = (row.get("evidence") or "").strip()

            if current_evidence == original_quote:
                quote_status = "ALREADY_EXACT"
            else:
                # Restore only a quote verified against the original chunk.
                row["evidence"] = original_quote
                quote_status = "RESTORED_FROM_ORIGINAL"

        # Track edits as information, not as automatic errors.
        question_status = "NOT_CHECKED"
        answer_status = "NOT_CHECKED"

        if original is not None:
            original_question = (original.get("question") or "").strip()
            original_answer = (
                original.get("answer")
                or original.get("gold_answer")
                or ""
            ).strip()

            reviewed_question = (row.get("question") or "").strip()
            reviewed_answer = (row.get("gold_answer") or "").strip()

            question_status = (
                "UNCHANGED"
                if reviewed_question == original_question
                else "EDITED"
            )
            answer_status = (
                "UNCHANGED"
                if reviewed_answer == original_answer
                else "EDITED"
            )

        if not (row.get("question") or "").strip():
            issues.append("missing_question")

        if not (row.get("gold_answer") or "").strip():
            issues.append("missing_gold_answer")

        if not (row.get("evidence") or "").strip():
            issues.append("missing_evidence")

        # Check the review row's source metadata against the original.
        if original is not None:
            original_unit = original.get("unit_id")
            reviewed_unit = row.get("evidence_unit")

            if (
                original_unit is not None
                and reviewed_unit
                and str(original_unit) != str(reviewed_unit)
            ):
                issues.append("evidence_unit_mismatch")

            original_position = original.get("reveal_position")
            reviewed_position = row.get("reveal_position")

            if (
                original_position is not None
                and reviewed_position
                and str(original_position) != str(reviewed_position)
            ):
                issues.append("reveal_position_mismatch")

        # Edited wording is allowed. Genuine integrity issues are not.
        status = "CHECK" if issues else "PASS"

        output_rows.append(row)
        report_rows.append({
            "csv_row": row_number,
            "id": question_id,
            "chunk_id": chunk_id,
            "review_status": row.get("review_status", ""),
            "validation_status": status,
            "question_status": question_status,
            "answer_status": answer_status,
            "evidence_status": quote_status,
            "issues": "; ".join(issues),
        })

    # Preserve the existing columns and add a status column if needed.
    output_fields = list(fieldnames)
    if "evidence_restore_status" not in output_fields:
        output_fields.append("evidence_restore_status")

    report_by_id = {}
    for report in report_rows:
        report_by_id.setdefault(report["id"], []).append(report)

    for row in output_rows:
        matching_reports = report_by_id.get(row.get("id", ""), [])
        if matching_reports:
            row["evidence_restore_status"] = matching_reports[0][
                "evidence_status"
            ]
        else:
            row["evidence_restore_status"] = "NOT_CHECKED"

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=output_fields,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(output_rows)

    with REPORT_FILE.open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        report_fields = [
            "csv_row",
            "id",
            "chunk_id",
            "review_status",
            "validation_status",
            "question_status",
            "answer_status",
            "evidence_status",
            "issues",
        ]
        writer = csv.DictWriter(file, fieldnames=report_fields)
        writer.writeheader()
        writer.writerows(report_rows)

    restored = sum(
        report["evidence_status"] == "RESTORED_FROM_ORIGINAL"
        for report in report_rows
    )
    already_exact = sum(
        report["evidence_status"] == "ALREADY_EXACT"
        for report in report_rows
    )
    needs_check = sum(
        report["validation_status"] == "CHECK"
        for report in report_rows
    )

    print("\nEvidence restoration finished.")
    print(f"Review rows processed: {len(output_rows)}")
    print(f"Quotes restored: {restored}")
    print(f"Quotes already exact: {already_exact}")
    print(f"Rows needing checks: {needs_check}")
    print(f"\nCorrected CSV: {OUTPUT_FILE}")
    print(f"Detailed report: {REPORT_FILE}")
    print("\nYour original reviewed CSV was not modified.")


if __name__ == "__main__":
    main()
