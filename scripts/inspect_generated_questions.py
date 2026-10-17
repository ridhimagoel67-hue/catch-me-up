
import json
from pathlib import Path

path = Path("data/processed/generated_questions.jsonl")

with path.open("r", encoding="utf-8") as file:
    records = [json.loads(line) for line in file if line.strip()]

print("Total records:", len(records))
print("Unique chunk IDs:", len({r["chunk_id"] for r in records}))
print("Episode positions:", sorted({r["reveal_position"] for r in records}))

print("\n" + "=" * 70)

for record in records:
    print("\nChunk:", record["chunk_id"])
    print("Episode:", record["unit_id"])
    print("Question:", record["question"])
    print("Answer:", record["answer"])
    print("Evidence:", record["support_quote"])
    print("-" * 70)
