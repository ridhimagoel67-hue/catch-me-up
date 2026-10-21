
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

# NEW: Evidence chunk whose retrieval rank we want to inspect.
TARGET_EVIDENCE = {
    "q02": "s01e01-c01",
}


def find_rank(hits, predicate):
    """Return the 1-based rank of the first matching chunk, or None."""
    return next(
        (
            rank
            for rank, (chunk, score) in enumerate(hits, start=1)
            if predicate(chunk)
        ),
        None,
    )


def evaluate_method(retriever, questions, quotes, method_name, method):
    """Evaluate retrieval and inspect specifically targeted evidence."""

    evaluated_ranks = []
    missing_quotes = []
    delayed_excluded = []
    leak_count = 0
    target_ranks = {}

    print("\n" + "=" * 65)
    print(f"{method_name.upper()} RETRIEVAL")
    print("=" * 65)

    search_function = getattr(retriever, method)

    for _, row in questions.iterrows():
        question_id = str(row["id"])
        question = str(row["question"])
        asked_at = int(row["asked_at"])
        reveal_position = int(row["reveal_position"])

        # Retrieve evidence at the position when the question is asked.
        hits = search_function(
            question,
            asked_at,
            k=SEARCH_K,
        )

        # Check all questions for future-content leakage.
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

        # NEW: Rank the exact target chunk independently of quote matching.
        target_chunk_id = TARGET_EVIDENCE.get(question_id)

        if target_chunk_id:
            target_rank = find_rank(
                hits,
                lambda chunk: chunk.chunk_id == target_chunk_id,
            )
            target_ranks[question_id] = target_rank

            if target_rank is None:
                print(
                    f"  Target evidence {target_chunk_id}: "
                    f"NOT FOUND in top {SEARCH_K}"
                )
            else:
                print(
                    f"  Target evidence {target_chunk_id}: "
                    f"rank {target_rank}"
                )

        # Do not score evidence that had not been revealed yet.
        if reveal_position > asked_at:
            delayed_excluded.append(question_id)
            print(
                "  Ordinary retrieval metric: EXCLUDED "
                "(evidence reveal position is later)"
            )
            continue

        quote = quotes.get(question_id)

        if not isinstance(quote, str) or not quote.strip():
            missing_quotes.append(question_id)
            print("  Gold evidence rank: NOT EVALUATED (no gold quote)")
            continue

        quote = quote.strip().casefold()

        quote_rank = find_rank(
            hits,
            lambda chunk: quote in chunk.text.casefold(),
        )

        evaluated_ranks.append(quote_rank)

        if quote_rank is None:
            print(
                f"  Gold evidence rank: not found in top {SEARCH_K}"
            )
        else:
            print(f"  Gold evidence rank: {quote_rank}")

    total = len(evaluated_ranks)
    rank_1 = sum(rank == 1 for rank in evaluated_ranks)
    recall_k = sum(
        rank is not None and rank <= K
        for rank in evaluated_ranks
    )
    recall_search_k = sum(
        rank is not None and rank <= SEARCH_K
        for rank in evaluated_ranks
    )

    results = {
        "method": method_name,
        "evaluated": total,
        "delayed_excluded": len(delayed_excluded),
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
        # NEW: Keep q02's exact evidence rank for the final comparison.
        "q02_target_rank": target_ranks.get("q02"),
    }

    print(f"\n{method_name} summary")
    print(f"Questions with gold quotes evaluated: {total}")
    print(f"Delayed questions excluded: {len(delayed_excluded)}")
    print(f"Eligible questions missing gold quotes: {len(missing_quotes)}")
    print(f"Future-content leaks: {leak_count}")
    print(f"Evidence ranked #1: {rank_1}/{total}")
    print(f"Recall@{K}: {recall_k}/{total}")
    print(f"Recall@{SEARCH_K}: {recall_search_k}/{total}")
    print(f"Top-1 rate: {results['top1_rate']:.3f}")
    print(f"Recall@{K}: {results['recall_at_k']:.3f}")
    print(f"Recall@{SEARCH_K}: {results['recall_at_search_k']:.3f}")

    if delayed_excluded:
        print("Delayed questions:", ", ".join(delayed_excluded))

    if missing_quotes:
        print(
            "Eligible questions missing quotes:",
            ", ".join(missing_quotes),
        )

    return results


def main():
    # 1. Validate required input files.
    required_files = [DATA, QUESTIONS, QUOTES]
    missing_files = [
        str(path) for path in required_files if not path.is_file()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Required evaluation files are missing:\n"
            + "\n".join(missing_files)
        )

    # 2. Load the content and build one shared retriever.
    units = json.loads(DATA.read_text(encoding="utf-8"))
    chunks = chunk_all(units)

    if not chunks:
        raise ValueError("No chunks were created from the input content.")

    print(f"Loaded {len(units)} units and {len(chunks)} chunks.")

    # NEW: Confirm the target evidence actually exists in the dataset.
    chunk_ids = {chunk.chunk_id for chunk in chunks}

    for question_id, target_id in TARGET_EVIDENCE.items():
        if target_id not in chunk_ids:
            raise ValueError(
                f"Target chunk {target_id} for {question_id} "
                "does not exist in the loaded dataset."
            )

    retriever = Retriever(chunks, embed)

    # 3. Load benchmark questions and gold evidence quotes.
    questions = pd.read_csv(QUESTIONS, dtype={"id": str})
    quote_table = pd.read_csv(QUOTES, dtype={"id": str})

    required_question_columns = {
        "id",
        "asked_at",
        "reveal_position",
        "question",
    }
    missing_columns = required_question_columns - set(questions.columns)

    if missing_columns:
        raise ValueError(
            f"{QUESTIONS} is missing columns: {sorted(missing_columns)}"
        )

    if questions["id"].duplicated().any():
        duplicates = questions.loc[
            questions["id"].duplicated(), "id"
        ].tolist()
        raise ValueError(f"Duplicate question IDs: {duplicates}")

    if not {"id", "quote"}.issubset(quote_table.columns):
        raise ValueError(f"{QUOTES} must contain 'id' and 'quote' columns.")

    if quote_table["id"].duplicated().any():
        duplicates = quote_table.loc[
            quote_table["id"].duplicated(), "id"
        ].tolist()
        raise ValueError(f"Duplicate quote IDs: {duplicates}")

    quotes = dict(zip(quote_table["id"], quote_table["quote"]))

    # 4. Compare every retrieval method using identical inputs.
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

    # 5. Print the standard metrics and q02's target rank.
    print("\n" + "=" * 65)
    print("FINAL RETRIEVAL COMPARISON")
    print("=" * 65)

    comparison = pd.DataFrame(all_results)

    columns = [
        "method",
        "evaluated",
        "delayed_excluded",
        "missing_quotes",
        "top1_rate",
        "recall_at_k",
        "recall_at_search_k",
        "q02_target_rank",
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

    # 6. Fail if any method retrieved future content.
    total_leaks = sum(result["leaks"] for result in all_results)

    if total_leaks:
        raise AssertionError(
            f"Future-content leakage detected across methods: "
            f"{total_leaks} leak(s)."
        )

    print("\nAll three methods passed the position-leak check.")
    print(
        "Reminder: retrieval scores use exact gold-quote matching "
        "on the currently annotated benchmark. Results are based "
        "on a small evaluation set."
    )


if __name__ == "__main__":
    main()
