import json
from pathlib import Path

import pandas as pd

from catchmeup.engine.chunking import chunk_all
from catchmeup.engine.embedding import embed
from catchmeup.engine.retriever import Retriever

DATA = Path("data/processed/gossip-girl_s1.json")
QUESTIONS = Path("data/benchmark/questions_gg_s1.csv")
QUOTES = Path("data/benchmark/evidence_quotes.csv")
K = 3

units = json.loads(DATA.read_text(encoding="utf-8"))
chunks = chunk_all(units)
sizes = [len(c.text.split()) for c in chunks]
print(f"{len(units)} episodes -> {len(chunks)} chunks "
      f"(average {sum(sizes) / len(sizes):.0f} words, longest {max(sizes)})")

retriever = Retriever(chunks, embed)
questions = pd.read_csv(QUESTIONS)
quote_table = pd.read_csv(QUOTES)
quotes = dict(zip(quote_table["id"], quote_table["quote"]))

ranks = []
for _, q in questions.iterrows():
    asked_at = int(q["asked_at"])
    hits = retriever.search(q["question"], asked_at, k=10)
    assert all(c.position <= asked_at for c, _ in hits), "LEAK: future chunk returned!"
    print(f"\n{q['id']} [{q['type']}] viewer at episode {asked_at}: {q['question']}")
    for chunk, score in hits[:K]:
        print(f"   {score:.2f}  {chunk.label:<6} {chunk.text[:80]!r}")
    if q["id"] in quotes:
        quote = quotes[q["id"]].lower()
        rank = next((i for i, (c, _) in enumerate(hits, start=1)
                     if quote in c.text.lower()), None)
        ranks.append(rank)
        print("   answer sentence found at rank:", rank if rank else "not in top 10")

n = len(ranks)
print(f"\nLeak check passed for all {len(questions)} questions.")
print(f"Answer passage ranked #1: {sum(r == 1 for r in ranks)} of {n}")
print(f"Answer passage in top {K}: {sum(r is not None and r <= K for r in ranks)} of {n}")