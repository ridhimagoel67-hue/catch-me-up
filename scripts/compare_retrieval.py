
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
SEARCH_K = 10

METHODS = {
    "Vector": "search",
    "BM25": "search_bm25",
    "Hybrid": "search_hybrid",
}


def evaluate_method(retriever, questions, quotes, method_name, method):
    evaluated_ranks = []
    missing_quotes = []
    leak_count = 0

    print("\n" + "=" * 65)
    print(f"{method_name.upper()} RETRIEVAL")
    print("=" * 65)

    for _, row in questions.iterrows():
        question_id = row["id"]
        question = row["question"]
        asked_at = int(row["asked_at"])

        search_function = getattr(retriever, method)

        hits = search_function(
            question,
            asked_at,
            k=SEARCH_K,
        )

        has_leak = any(
            chunk.position > asked_at
            for chunk, score in hits
        )

        if has_leak:
            leak_count += 1

        print(
            f"\n{question_id} | position {asked_at} | "
            f"{question}"
        )

        for chunk, score in hits[:K]:
            print(
                f"  {score:.4f} | {chunk.chunk_id} | "
                f"{chunk.label} | {chunk.text[:100]!r}"
            )

        quote = quotes.get(question_id)

        if not isinstance(quote, str) or not quote.strip():
            missing_quotes.append(question_id)
            print("  Gold evidence rank: NOT EVALUATED")
            continue

        quote = quote.strip().casefold()

        rank = next(
            (
                i
                for i, (chunk, score) in enumerate(hits, start=1)
                if quote in chunk.text.casefold()
            ),
            None,
        )

        evaluated_ranks.append(rank)

        if rank is None:
            print(f"  Gold evidence rank: not found in top {SEARCH_K}")
        else:
            print(f"  Gold evidence rank: {rank}")

    total = len(evaluated_ranks)
    rank_1 = sum(rank == 1 for rank in evaluated_ranks)
    recall_k = sum(
        rank is not None and rank <= K
        for rank in evaluated_ranks
    )
    recall_search_k = sum(
        rank is not None
        for rank in evaluated_ranks
    )

    results = {
        "method": method_name,
        "evaluated": total,
        "missing_quotes": len(missing_quotes),
        "leaks": leak_count,
        "rank_1": rank_1,
        "recall_k": recall_k,
        "recall_search_k": recall_search_k,
        "top1_rate": rank_1 / total if total else 0,
        "recall_at_k": recall_k / total if total else 0,
        "recall_at_search_k": (
            recall_search_k / total if total else 0
        ),
    }

    print(f"\n{method_name} summary")
    print(f"Questions with gold quotes: {total}")
    print(f"Questions without gold quotes: {len(missing_quotes)}")
    print(f"Future-content leaks: {leak_count}")
    print(f"Evidence ranked #1: {rank_1}/{total}")
    print(f"Recall@{K}: {recall_k}/{total}")
    print(f"Recall@{SEARCH_K}: {recall_search_k}/{total}")
    print(f"Top-1 rate: {results['top1_rate']:.3f}")
    print(f"Recall@{K}: {results['recall_at_k']:.3f}")
    print(f"Recall@{SEARCH_K}: {results['recall_at_search_k']:.3f}")

    return results


def main():
    units = json.loads(DATA.read_text(encoding="utf-8"))
    chunks = chunk_all(units)

    print(f"Loaded {len(units)} units and {len(chunks)} chunks.")

    # Build one retriever so all methods use identical chunks and embeddings.
    retriever = Retriever(chunks, embed)

    questions = pd.read_csv(QUESTIONS)
    quote_table = pd.read_csv(QUOTES)

    quotes = dict(zip(quote_table["id"], quote_table["quote"]))

    all_results = []

    for method_name, method in METHODS.items():
        result = evaluate_method(
            retriever,
            questions,
            quotes,
            method_name,
            method,
        )
        all_results.append(result)

    print("\n" + "=" * 65)
    print("FINAL RETRIEVAL COMPARISON")
    print("=" * 65)

    comparison = pd.DataFrame(all_results)

    columns = [
        "method",
        "evaluated",
        "top1_rate",
        "recall_at_k",
        "recall_at_search_k",
        "leaks",
    ]

    print(
        comparison[columns].to_string(
            index=False,
            formatters={
                "top1_rate": "{:.3f}".format,
                "recall_at_k": "{:.3f}".format,
                "recall_at_search_k": "{:.3f}".format,
            },
        )
    )

    total_leaks = sum(result["leaks"] for result in all_results)

    if total_leaks:
        raise AssertionError(
            f"Future-content leakage detected across methods: "
            f"{total_leaks} leak(s)."
        )

    print("\nAll three methods passed the position-leak check.")


if __name__ == "__main__":
    main()
