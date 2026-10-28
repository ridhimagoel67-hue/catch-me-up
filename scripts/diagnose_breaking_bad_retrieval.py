
from catchmeup.collections import get_collection
from catchmeup.engine.chunking import chunk_all
from catchmeup.engine.embedding import embed
from catchmeup.engine.retriever import Retriever
from scripts.test_rag import load_units

config = get_collection("breaking-bad-full")
units = load_units(config.data_path)
chunks = chunk_all(units)
retriever = Retriever(chunks, embed)

tests = [
    ("Season 1, Episode 1: Pilot", 1),
    ("What is the title of Season 1 Episode 2?", 1),
    ("What is the title of Season 1 Episode 2?", 2),
]

for question, position in tests:
    print(f"\nQuestion: {question}")
    print(f"Current position: {position}")

    results = retriever.search_hybrid(question, position, k=5)

    if not results:
        print("No results returned.")
        continue

    for chunk, score in results:
        print(
            f"score={score:.4f} | "
            f"position={chunk.position} | "
            f"id={chunk.chunk_id} | "
            f"text={chunk.text}"
        )
