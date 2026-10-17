
import json
from collections import Counter
from pathlib import Path

path = Path("data/processed/generated_questions.jsonl")

records = []

with path.open("r", encoding="utf-8") as file:
    for line_number, line in enumerate(file, start=1):
        if not line.strip():
            continue

        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            print(f"Malformed JSON on line {line_number}")

print("Total records:", len(records))

chunk_ids = [r.get("chunk_id") for r in records]
print("Unique chunk IDs:", len(set(chunk_ids)))

print("\nRecords by episode:")
episode_counts = Counter(r.get("unit_id") for r in records)

for episode, count in sorted(episode_counts.items()):
    print(f"  {episode}: {count}")

print("\nSuitability:")
suitability_counts = Counter(
    str(r.get("suitable", "legacy"))
    for r in records
)
for status, count in sorted(suitability_counts.items()):
    print(f"  {status}: {count}")

print("\nPotential issues:")

issues = 0

for record in records:
    chunk_id = record.get("chunk_id", "unknown")

    if record.get("suitable") is False:
        continue

    for field in ("question", "answer", "support_quote"):
        if not isinstance(record.get(field), str) or not record[field].strip():
            print(f"  {chunk_id}: missing or empty {field}")
            issues += 1

    if record.get("question", "").strip().endswith("?") is False:
        print(f"  {chunk_id}: question may be missing a question mark")
        issues += 1

print("\nPotential issues found:", issues)
