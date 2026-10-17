
import csv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

SOURCE_FILE = (
    BASE_DIR
    / "data"
    / "benchmark"
    / "questions_verified_against_source.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "benchmark"
    / "questions_gg_s1_candidates.csv"
)


def get_episode_position(evidence_unit):
    """Convert s01e02 into episode position 2."""
    if not evidence_unit:
        return None

    try:
        return int(evidence_unit.split("e")[1])
    except (IndexError, ValueError):
        return None


def main():
    if not SOURCE_FILE.exists():
        raise FileNotFoundError(
            f"Could not find {SOURCE_FILE}\n"
            "Check that the verified CSV is in data/benchmark/."
        )

    with SOURCE_FILE.open(
        "r", newline="", encoding="utf-8-sig"
    ) as file:
        questions = list(csv.DictReader(file))

    candidates = []

    for question in questions:
        episode = get_episode_position(
            question.get("evidence_unit", "")
        )

        # Episode 1 has no earlier episode in this dataset
        # for a before-reveal test.
        if episode is None or episode < 2:
            continue

        # These are candidates, not confirmed reveal positions.
        for test_type, asked_at in [
            ("A", episode),
            ("B", episode - 1),
        ]:
            candidates.append({
                "id": f'{question["id"]}-{test_type}',
                "type": test_type,
                "asked_at": asked_at,
                "reveal_position": episode,
                "evidence_unit": question["evidence_unit"],
                "question": question["question"],
                "gold_answer": question["gold_answer"],
                "evidence": question["evidence"],
                "metadata_status": "needs_reveal_review",
            })

    fieldnames = [
        "id",
        "type",
        "asked_at",
        "reveal_position",
        "evidence_unit",
        "question",
        "gold_answer",
        "evidence",
        "metadata_status",
    ]

    with OUTPUT_FILE.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(candidates)

    print(f"Source questions: {len(questions)}")
    print(f"Candidate test cases: {len(candidates)}")
    print(f"Saved to: {OUTPUT_FILE}")
    print("Reminder: review reveal positions before evaluation.")


if __name__ == "__main__":
    main()
