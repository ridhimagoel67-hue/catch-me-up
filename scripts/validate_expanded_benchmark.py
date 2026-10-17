
import csv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

CANDIDATE_FILE = (
    BASE_DIR / "data" / "benchmark"
    / "questions_gg_s1_candidates.csv"
)

SOURCE_FILE = (
    BASE_DIR / "data" / "processed"
    / "gossip-girl_s1.json"
)


def main():
    with CANDIDATE_FILE.open(
        encoding="utf-8-sig", newline=""
    ) as file:
        rows = list(csv.DictReader(file))

    ids = [row["id"] for row in rows]
    duplicate_ids = len(ids) - len(set(ids))

    invalid_positions = []
    invalid_types = []
    invalid_evidence = []

    import json

    with SOURCE_FILE.open(encoding="utf-8-sig") as file:
        episodes = json.load(file)

    source_by_unit = {
        episode["unit_id"]: episode["text"]
        for episode in episodes
    }

    for row in rows:
        try:
            asked_at = int(row["asked_at"])
            reveal_position = int(row["reveal_position"])
        except ValueError:
            invalid_positions.append(row["id"])
            continue

        if asked_at > reveal_position or asked_at < 1:
            invalid_positions.append(row["id"])

        if row["type"] not in {"A", "B"}:
            invalid_types.append(row["id"])

        source_text = source_by_unit.get(row["evidence_unit"])
        if source_text is None or row["evidence"] not in source_text:
            invalid_evidence.append(row["id"])

    print("Total rows:", len(rows))
    print("Duplicate IDs:", duplicate_ids)
    print("Invalid positions:", invalid_positions)
    print("Invalid types:", invalid_types)
    print("Evidence mismatches:", invalid_evidence)


if __name__ == "__main__":
    main()
