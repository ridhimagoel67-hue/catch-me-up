import json
from pathlib import Path

import pandas as pd

from catchmeup.engine.chunking import chunk_all
from catchmeup.engine.embedding import embed
from catchmeup.engine.retriever import Retriever


DATA = Path("data/processed/gossip-girl_s1.json")
QUESTIONS = Path("data/benchmark/questions_gg_s1.csv")
QUOTES = Path("data/benchmark/evidence_quotes.csv")
DELAYED_QUOTES = Path("data/benchmark/delayed_evidence.csv")

K = 3
SEARCH_K = 10


def find_quote_rank(hits, quote):
    """Return the 1-based rank of the chunk containing the gold quote."""
    quote = str(quote).strip().casefold()

    if not quote:
        return None

    for rank, (chunk, score) in enumerate(hits, start=1):
        if quote in chunk.text.casefold():
            return rank

    return None


def quote_exists_by_position(chunks, quote, position):
    """Check whether the exact gold quote exists in content available by position."""
    quote = str(quote).strip().casefold()

    if not quote:
        return False

    return any(
        chunk.position <= position and quote in chunk.text.casefold()
        for chunk in chunks
    )


def format_rank(rank):
    """Format a retrieval rank for readable output."""
    return f"rank {rank}" if rank is not None else f"not found in top {SEARCH_K}"


def print_metrics(title, ranks):
    """Print ranking metrics for one evaluation group."""
    total = len(ranks)

    print("\n" + "=" * 50)
    print(title)
    print("=" * 50)
    print(f"Gold quotes evaluated: {total}")

    if total == 0:
        print("No gold quotes available.")
        return

    rank_1 = sum(rank == 1 for rank in ranks)
    in_top_k = sum(rank is not None and rank <= K for rank in ranks)
    in_search_k = sum(
        rank is not None and rank <= SEARCH_K
        for rank in ranks
    )

    print(f"Evidence ranked #1: {rank_1}/{total}")
    print(f"Evidence in top {K}: {in_top_k}/{total}")
    print(f"Evidence found in top {SEARCH_K}: {in_search_k}/{total}")
    print(f"Top-1 rate: {rank_1 / total:.3f}")
    print(f"Recall@{K}: {in_top_k / total:.3f}")
    print(f"Recall@{SEARCH_K}: {in_search_k / total:.3f}")


def main():
    # 1. Validate required files.
    required_files = [DATA, QUESTIONS, QUOTES, DELAYED_QUOTES]
    missing_files = [str(path) for path in required_files if not path.exists()]

    if missing_files:
        raise FileNotFoundError(
            "Required evaluation files are missing:\n"
            + "\n".join(missing_files)
        )

    # 2. Load the content and build chunks.
    units = json.loads(DATA.read_text(encoding="utf-8"))
    chunks = chunk_all(units)

    if not chunks:
        raise ValueError("No chunks were created from the input content.")

    sizes = [len(chunk.text.split()) for chunk in chunks]

    print(
        f"{len(units)} units -> {len(chunks)} chunks "
        f"(average {sum(sizes) / len(sizes):.0f} words, "
        f"longest {max(sizes)})"
    )

    retriever = Retriever(chunks, embed)

    # 3. Load benchmark questions and the two independent evidence files.
    questions = pd.read_csv(QUESTIONS, dtype={"id": str})
    quote_table = pd.read_csv(QUOTES, dtype={"id": str})
    delayed_table = pd.read_csv(DELAYED_QUOTES, dtype={"id": str})

    required_question_columns = {
        "id",
        "type",
        "asked_at",
        "reveal_position",
        "question",
    }
    missing_columns = required_question_columns - set(questions.columns)

    if missing_columns:
        raise ValueError(
            f"{QUESTIONS} is missing columns: {sorted(missing_columns)}"
        )

    for table, path in [
        (quote_table, QUOTES),
        (delayed_table, DELAYED_QUOTES),
    ]:
        if not {"id", "quote"}.issubset(table.columns):
            raise ValueError(f"{path} must contain 'id' and 'quote' columns.")

    if questions["id"].duplicated().any():
        duplicate_ids = questions.loc[
            questions["id"].duplicated(), "id"
        ].tolist()
        raise ValueError(f"Duplicate question IDs: {duplicate_ids}")

    questions_by_id = questions.set_index("id", drop=False)
    quotes = dict(zip(quote_table["id"], quote_table["quote"]))

    # Normal retrieval metrics and delayed-evidence metrics stay separate.
    normal_ranks = []
    missing_quotes = []
    asking_position_leaks = 0
    delayed_position_leaks = 0

    # 4. Evaluate ordinary retrieval at each question's asking position.
    print("\nEvaluating questions at their asking positions...")

    for _, row in questions.iterrows():
        question_id = row["id"]
        question = str(row["question"])
        asked_at = int(row["asked_at"])
        reveal_position = int(row["reveal_position"])

        quote = quotes.get(question_id)
        has_quote = isinstance(quote, str) and bool(quote.strip())

        hits = retriever.search(question, asked_at, k=SEARCH_K)

        has_leak = any(
            chunk.position > asked_at
            for chunk, score in hits
        )

        if has_leak:
            asking_position_leaks += 1

        print(
            f"\n{question_id} [{row['type']}] "
            f"position {asked_at}: {question}"
        )

        for chunk, score in hits[:K]:
            print(
                f"  {score:.3f} | {chunk.chunk_id} | "
                f"{chunk.label} | {chunk.text[:100]!r}"
            )

        # A quote whose reveal position is later than asked_at belongs
        # in delayed evaluation, not in ordinary retrieval metrics.
        if reveal_position > asked_at:
            print("  Ordinary metric: excluded (delayed evidence)")
            continue

        if not has_quote:
            missing_quotes.append(question_id)
            print("  Evidence rank: NOT EVALUATED (no gold quote)")
            continue

        rank = find_quote_rank(hits, quote)
        normal_ranks.append(rank)
        print(f"  Evidence rank: {format_rank(rank)}")

    print_metrics("NORMAL RETRIEVAL SUMMARY", normal_ranks)

    # 5. Evaluate each row in delayed_evidence.csv independently.
    print("\n" + "=" * 50)
    print("DELAYED-EVIDENCE EVALUATION")
    print("=" * 50)

    delayed_results = []
    missing_question_ids = []

    for _, delayed_row in delayed_table.iterrows():
        question_id = delayed_row["id"]
        quote = delayed_row["quote"]

        if not isinstance(quote, str) or not quote.strip():
            print(f"\n{question_id}: skipped — gold quote is empty.")
            continue

        if question_id not in questions_by_id.index:
            missing_question_ids.append(question_id)
            print(
                f"\n{question_id}: skipped — ID does not exist "
                "in the questions benchmark."
            )
            continue

        row = questions_by_id.loc[question_id]

        if isinstance(row, pd.DataFrame):
            raise ValueError(
                f"Question ID {question_id!r} matched multiple rows."
            )

        question = str(row["question"])
        asked_at = int(row["asked_at"])
        reveal_position = int(row["reveal_position"])

        if reveal_position <= asked_at:
            print(
                f"\n{question_id}: warning — reveal position "
                f"({reveal_position}) is not later than asking position "
                f"({asked_at}). Check the benchmark annotation."
            )

        # Search at the original asking position.
        asking_hits = retriever.search(
            question,
            asked_at,
            k=SEARCH_K,
        )
        asking_rank = find_quote_rank(asking_hits, quote)

        asking_leak = any(
            chunk.position > asked_at
            for chunk, score in asking_hits
        )
        if asking_leak:
            asking_position_leaks += 1

        # Search again at the evidence's annotated reveal position.
        reveal_hits = retriever.search(
            question,
            reveal_position,
            k=SEARCH_K,
        )
        reveal_rank = find_quote_rank(reveal_hits, quote)

        reveal_leak = any(
            chunk.position > reveal_position
            for chunk, score in reveal_hits
        )
        if reveal_leak:
            delayed_position_leaks += 1

        quote_available = quote_exists_by_position(
            chunks,
            quote,
            reveal_position,
        )

        print(f"\n{question_id}: {question}")
        print(f"  Asked at position: {asked_at}")
        print(f"  Reveal position: {reveal_position}")
        print(f"  Gold quote available by reveal position: {quote_available}")
        print(f"  At asking position: {format_rank(asking_rank)}")
        print(f"  At reveal position: {format_rank(reveal_rank)}")

        delayed_results.append(
            {
                "id": question_id,
                "asking_rank": asking_rank,
                "reveal_rank": reveal_rank,
                "quote_available": quote_available,
            }
        )

    # 6. Summarize delayed-evidence retrieval separately.
    print("\n" + "=" * 50)
    print("DELAYED-EVIDENCE SUMMARY")
    print("=" * 50)
    print(f"Delayed annotations evaluated: {len(delayed_results)}")

    if delayed_results:
        available_count = sum(
            result["quote_available"] for result in delayed_results
        )
        found_at_asking = sum(
            result["asking_rank"] is not None
            for result in delayed_results
        )
        found_at_reveal = sum(
            result["reveal_rank"] is not None
            for result in delayed_results
        )

        print(
            "Gold quotes present by their reveal positions: "
            f"{available_count}/{len(delayed_results)}"
        )
        print(
            f"Quotes retrieved at asking positions: "
            f"{found_at_asking}/{len(delayed_results)}"
        )
        print(
            f"Quotes retrieved at reveal positions: "
            f"{found_at_reveal}/{len(delayed_results)}"
        )

    # 7. Report annotation coverage and position-filter guardrails.
    print("\n" + "=" * 50)
    print("EVALUATION CHECKS")
    print("=" * 50)
    print(f"Questions in benchmark: {len(questions)}")
    print(f"Questions with ordinary gold quotes: {len(quotes)}")
    print(f"Questions without ordinary gold quotes: {len(missing_quotes)}")
    print(f"Delayed evidence annotations: {len(delayed_table)}")
    print(f"Unknown IDs in delayed evidence: {len(missing_question_ids)}")
    print(f"Future-content leaks at asking positions: {asking_position_leaks}")
    print(f"Future-content leaks at reveal positions: {delayed_position_leaks}")

    if missing_quotes:
        print("Questions needing ordinary evidence quotes:")
        print(", ".join(missing_quotes))

    if missing_question_ids:
        print("Delayed evidence IDs absent from the question benchmark:")
        print(", ".join(missing_question_ids))

    if asking_position_leaks > 0 or delayed_position_leaks > 0:
        raise AssertionError(
            "Future-content leakage detected. "
            f"Asking positions: {asking_position_leaks}; "
            f"reveal positions: {delayed_position_leaks}."
        )


if __name__ == "__main__":
    main()
