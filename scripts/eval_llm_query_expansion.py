
import json
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from catchmeup.engine.chunking import chunk_all
from catchmeup.engine.embedding import embed
from catchmeup.engine.retriever import Retriever


DATA = Path("data/processed/gossip-girl_s1.json")
QUESTIONS = Path("data/benchmark/questions_gg_s1.csv")
QUOTES = Path("data/benchmark/evidence_quotes.csv")

SEARCH_K = 10
K = 3


def create_client():
    """Load the existing LLM configuration from .env."""
    load_dotenv()

    api_key = os.getenv("LLM_API_KEY")
    base_url = os.getenv("LLM_BASE_URL")
    model = os.getenv("LLM_MODEL")

    if not api_key or not base_url or not model:
        raise ValueError(
            "Missing LLM_API_KEY, LLM_BASE_URL, or LLM_MODEL in .env"
        )

    client = OpenAI(api_key=api_key, base_url=base_url)
    return client, model


def expand_query(client, model, question):
    """Generate a concise search query without answering the question."""
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "Rewrite the user's question as one concise search query "
                    "for retrieving relevant passages from a collection of "
                    "episode summaries. Preserve the original meaning. "
                    "Include useful names, relationships, events, and "
                    "synonyms when appropriate. Do not answer the question. "
                    "Do not assume facts that are not stated in the question. "
                    "Return only the search query."
                ),
            },
            {
                "role": "user",
                "content": question,
            },
        ],
    )

    query = response.choices[0].message.content

    if not query or not query.strip():
        raise ValueError(f"LLM returned an empty query for: {question}")

    return query.strip()


def find_gold_rank(hits, quote):
    """Find the rank of the chunk containing the exact gold quote."""
    if not isinstance(quote, str) or not quote.strip():
        return None

    quote = quote.strip().casefold()

    for rank, (chunk, score) in enumerate(hits, start=1):
        if quote in chunk.text.casefold():
            return rank

    return None


def evaluate_ranks(ranks):
    total = len(ranks)

    if total == 0:
        return {
            "total": 0,
            "top1": 0,
            "recall3": 0,
            "recall10": 0,
        }

    return {
        "total": total,
        "top1": sum(rank == 1 for rank in ranks),
        "recall3": sum(
            rank is not None and rank <= K for rank in ranks
        ),
        "recall10": sum(rank is not None for rank in ranks),
    }


def print_metrics(title, ranks, leaks):
    metrics = evaluate_ranks(ranks)
    total = metrics["total"]

    print(f"\n{title}")
    print("-" * len(title))
    print(f"Gold-quote questions: {total}")
    print(f"Future-content leaks: {leaks}")

    if total == 0:
        print("No gold quotes available for evaluation.")
        return

    print(
        f"Evidence at rank 1: "
        f"{metrics['top1']}/{total} "
        f"({metrics['top1'] / total:.3f})"
    )
    print(
        f"Recall@{K}: "
        f"{metrics['recall3']}/{total} "
        f"({metrics['recall3'] / total:.3f})"
    )
    print(
        f"Recall@{SEARCH_K}: "
        f"{metrics['recall10']}/{total} "
        f"({metrics['recall10'] / total:.3f})"
    )


def main():
    # 1. Load the dataset and create chunks.
    units = json.loads(DATA.read_text(encoding="utf-8"))
    chunks = chunk_all(units)

    print(f"Loaded {len(units)} units and {len(chunks)} chunks.")

    # 2. Initialize the retriever and LLM client.
    retriever = Retriever(chunks, embed)
    client, model = create_client()

    # 3. Load questions and gold evidence quotes.
    questions = pd.read_csv(QUESTIONS)
    quote_table = pd.read_csv(QUOTES)
    quotes = dict(zip(quote_table["id"], quote_table["quote"]))

    original_ranks = []
    expanded_ranks = []

    original_leaks = 0
    expanded_leaks = 0
    missing_quotes = []

    # 4. Compare the original and LLM-expanded query.
    for _, row in questions.iterrows():
        question_id = row["id"]
        question = row["question"]
        asked_at = int(row["asked_at"])

        print(f"\n{question_id}: {question}")

        # Generate a query, not an answer.
        try:
            expanded_query = expand_query(client, model, question)
        except Exception as exc:
            raise RuntimeError(
                f"Query expansion failed for {question_id}. "
                "Check your LLM configuration and connection."
            ) from exc

        print(f"Expanded query: {expanded_query}")

        original_hits = retriever.search(
            question, asked_at, k=SEARCH_K
        )
        expanded_hits = retriever.search(
            expanded_query, asked_at, k=SEARCH_K
        )

        # 5. Verify position restrictions.
        if any(
            chunk.position > asked_at
            for chunk, score in original_hits
        ):
            original_leaks += 1

        if any(
            chunk.position > asked_at
            for chunk, score in expanded_hits
        ):
            expanded_leaks += 1

        # 6. Find the evidence rank for each approach.
        quote = quotes.get(question_id)

        if not isinstance(quote, str) or not quote.strip():
            missing_quotes.append(question_id)
            print("Gold evidence: unavailable; skipped in metrics.")
            continue

        original_rank = find_gold_rank(original_hits, quote)
        expanded_rank = find_gold_rank(expanded_hits, quote)

        original_ranks.append(original_rank)
        expanded_ranks.append(expanded_rank)

        print(f"Original evidence rank: {original_rank}")
        print(f"Expanded evidence rank: {expanded_rank}")

    # 7. Report both approaches.
    print("\n" + "=" * 55)
    print("LLM QUERY EXPANSION EVALUATION")
    print("=" * 55)

    print(f"Questions in benchmark: {len(questions)}")
    print(f"Questions with gold quotes: {len(original_ranks)}")
    print(f"Questions without gold quotes: {len(missing_quotes)}")

    print_metrics(
        "ORIGINAL QUERY",
        original_ranks,
        original_leaks,
    )

    print_metrics(
        "LLM-EXPANDED QUERY",
        expanded_ranks,
        expanded_leaks,
    )

    if missing_quotes:
        print("\nQuestions needing gold quotes:")
        print(", ".join(str(qid) for qid in missing_quotes))

    if original_leaks or expanded_leaks:
        raise AssertionError(
            "Future-content leakage detected. Inspect the retriever."
        )


if __name__ == "__main__":
    main()
