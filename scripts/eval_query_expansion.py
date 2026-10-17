
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


def expand_query(question):
    """
    Simple, deterministic query expansion.

    Keep the original question and add useful content words
    as a second search phrase. This is a baseline experiment,
    not an LLM-generated rewrite.
    """
    stop_words = {
        "a", "an", "the", "is", "are", "was", "were",
        "do", "does", "did", "who", "what", "where",
        "when", "why", "how", "which", "and", "or",
        "to", "of", "in", "on", "for", "with", "between",
        "ever", "after", "being", "their", "she", "he",
        "they", "it", "her", "his", "them"
    }

    words = [
        word.strip(".,?!:;()[]\"'").casefold()
        for word in question.split()
    ]

    keywords = [
        word for word in words
        if word and word not in stop_words
    ]

    # Avoid repeating the same words.
    keywords = list(dict.fromkeys(keywords))

    if not keywords:
        return question

    return question + " " + " ".join(keywords)


def find_gold_rank(hits, quote):
    """Return the rank of the chunk containing the gold quote."""
    if not isinstance(quote, str) or not quote.strip():
        return None

    quote = quote.strip().casefold()

    return next(
        (
            rank
            for rank, (chunk, score) in enumerate(hits, start=1)
            if quote in chunk.text.casefold()
        ),
        None,
    )


def main():
    # 1. Load data and create chunks.
    units = json.loads(DATA.read_text(encoding="utf-8"))
    chunks = chunk_all(units)

    sizes = [len(chunk.text.split()) for chunk in chunks]

    print(
        f"{len(units)} units -> {len(chunks)} chunks "
        f"(average {sum(sizes) / len(sizes):.0f} words, "
        f"longest {max(sizes)})"
    )

    # 2. Create the retriever and load benchmark data.
    retriever = Retriever(chunks, embed)
    questions = pd.read_csv(QUESTIONS)
    quote_table = pd.read_csv(QUOTES)

    quotes = dict(zip(quote_table["id"], quote_table["quote"]))

    baseline_ranks = []
    expanded_ranks = []

    baseline_leaks = 0
    expanded_leaks = 0
    missing_quotes = []

    # 3. Compare original and expanded queries.
    for _, row in questions.iterrows():
        question_id = row["id"]
        question = row["question"]
        asked_at = int(row["asked_at"])

        expanded_query = expand_query(question)
        gold_quote = quotes.get(question_id)

        # Original query retrieval.
        baseline_hits = retriever.search(
            question,
            asked_at,
            k=SEARCH_K,
        )

        # Expanded query retrieval.
        expanded_hits = retriever.search(
            expanded_query,
            asked_at,
            k=SEARCH_K,
        )

        # 4. Check position restrictions for both methods.
        if any(
            chunk.position > asked_at
            for chunk, score in baseline_hits
        ):
            baseline_leaks += 1

        if any(
            chunk.position > asked_at
            for chunk, score in expanded_hits
        ):
            expanded_leaks += 1

        print(f"\n{question_id}: {question}")
        print(f"  Expanded query: {expanded_query}")

        # Show top 3 results for each method.
        print("  Original query top results:")
        for chunk, score in baseline_hits[:K]:
            print(
                f"    {score:.3f} | {chunk.chunk_id} | "
                f"{chunk.text[:90]!r}"
            )

        print("  Expanded query top results:")
        for chunk, score in expanded_hits[:K]:
            print(
                f"    {score:.3f} | {chunk.chunk_id} | "
                f"{chunk.text[:90]!r}"
            )

        # 5. Score only questions with gold quotes.
        if not isinstance(gold_quote, str) or not gold_quote.strip():
            missing_quotes.append(question_id)
            print("  Gold evidence: unavailable; not scored")
            continue

        baseline_rank = find_gold_rank(baseline_hits, gold_quote)
        expanded_rank = find_gold_rank(expanded_hits, gold_quote)

        baseline_ranks.append(baseline_rank)
        expanded_ranks.append(expanded_rank)

        print(
            f"  Gold evidence rank: "
            f"original={baseline_rank}, expanded={expanded_rank}"
        )

    # 6. Calculate metrics.
    def summarize(name, ranks, leaks):
        total = len(ranks)
        rank_1 = sum(rank == 1 for rank in ranks)
        top_k = sum(
            rank is not None and rank <= K
            for rank in ranks
        )
        top_search_k = sum(rank is not None for rank in ranks)

        print(f"\n{name}")
        print("-" * len(name))
        print(f"Future-content leaks: {leaks}")
        print(f"Evidence ranked #1: {rank_1}/{total}")
        print(f"Evidence in top {K}: {top_k}/{total}")
        print(f"Evidence in top {SEARCH_K}: {top_search_k}/{total}")

        if total:
            print(f"Top-1 rate: {rank_1 / total:.3f}")
            print(f"Recall@{K}: {top_k / total:.3f}")
            print(f"Recall@{SEARCH_K}: {top_search_k / total:.3f}")

    print("\n" + "=" * 55)
    print("QUERY EXPANSION EXPERIMENT")
    print("=" * 55)
    print(f"Questions: {len(questions)}")
    print(f"Questions with gold quotes: {len(baseline_ranks)}")
    print(f"Questions without gold quotes: {len(missing_quotes)}")

    summarize("ORIGINAL QUERY", baseline_ranks, baseline_leaks)
    summarize("EXPANDED QUERY", expanded_ranks, expanded_leaks)

    if missing_quotes:
        print("\nQuestions needing gold quotes:")
        print(", ".join(str(qid) for qid in missing_quotes))

    if baseline_leaks or expanded_leaks:
        raise AssertionError(
            "Future-content leakage detected. Inspect the retriever."
        )


if __name__ == "__main__":
    main()
