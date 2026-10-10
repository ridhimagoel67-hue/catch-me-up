import json
from pathlib import Path

import pandas as pd

from catchmeup.engine.chunking import chunk_all
from catchmeup.engine.embedding import embed
from catchmeup.engine.retriever import Retriever

units = json.loads(Path("data/processed/gossip-girl_s1.json").read_text(encoding="utf-8"))
chunks = chunk_all(units)
retriever = Retriever(chunks, embed)

questions = pd.read_csv("data/benchmark/questions_gg_s1.csv").set_index("id")
quote_table = pd.read_csv("data/benchmark/evidence_quotes.csv")
quote_table.columns = quote_table.columns.str.strip().str.lower()

for _, row in quote_table.iterrows():
    qid, quote = row["id"], str(row["quote"]).lower()
    asked_at = int(questions.loc[qid, "asked_at"])
    hits = retriever.search(questions.loc[qid, "question"], asked_at, k=len(chunks))
    holders = [(rank, c, s) for rank, (c, s) in enumerate(hits, start=1)
               if quote in c.text.lower()]
    print(f"\n{qid}  quote: {quote!r}")
    print(f"   {len(hits)} chunks were visible to the viewer")
    if not holders:
        print("   the quote is in NONE of the visible chunks (check its spelling)")
    for rank, c, s in holders:
        print(f"   found in {c.label}, rank {rank} of {len(hits)}, score {s:.2f}")
        print("   ", c.text[:300])
    print(f"   score of the #1 chunk: {hits[0][1]:.2f}")