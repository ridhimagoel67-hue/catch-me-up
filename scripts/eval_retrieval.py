import json
from pathlib import Path

import pandas as pd

from catchmeup.engine.chunking import chunk_all
from catchmeup.engine.embedding import embed
from catchmeup.engine.retriever import Retriever

DATA = Path("data/processed/gossip-girl_s1.json")
QUESTIONS = Path("data/benchmark/questions_gg_s1.csv")
K = 3

units = json.loads(DATA.read_text(encoding="utf-8"))
chunks = chunk_all(units)
print(f"{len(units)} episodes -> {len(chunks)} chunks")

retriever = Retriever(chunks, embed)
questions = pd.read_csv(QUESTIONS)

a_total = a_found = 0
for _, q in questions.iterrows():
    asked_at = int(q["asked_at"])
    hits = retriever.search(q["question"], asked_at, k=K)
    assert all(c.position <= asked_at for c, _ in hits), "LEAK: future chunk returned!"
    print(f"\n{q['id']} [{q['type']}] viewer at episode {asked_at}: {q['question']}")
    for chunk, score in hits:
        print(f"   {score:.2f}  {chunk.label:<6} {chunk.text[:80]!r}")
    if q["type"] == "A":
        found = q["evidence_unit"] in [c.unit_id for c, _ in hits]
        a_total += 1
        a_found += int(found)
        print("   right episode in top", K, ":", found)

print(f"\nLeak check passed for all {len(questions)} questions.")
print(f"Type A: right episode in top {K} for {a_found} of {a_total} questions.")