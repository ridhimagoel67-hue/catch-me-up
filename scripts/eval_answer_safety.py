
import pandas as pd
from pathlib import Path


QUESTIONS = Path("data/benchmark/questions_gg_s1.csv")

REFUSAL_PHRASE = (
    "I couldn't find enough evidence in the content available to you."
)


def expected_behavior(row):
    question_type = str(row["type"]).strip().upper()
    asked_at = int(row["asked_at"])
    reveal_position = int(row["reveal_position"])

    if question_type == "A":
        return "ANSWER"

    if question_type == "B":
        if asked_at < reveal_position:
            return "PROTECT_FUTURE"
        return "ANSWER_IF_SUPPORTED"

    if question_type == "C":
        return "OUTSIDE_CURRENT_COLLECTION"

    if question_type == "D":
        return "UNRESOLVED_MYSTERY"

    return "REVIEW_LABEL"


def main():
    if not QUESTIONS.is_file():
        raise FileNotFoundError(
            f"Benchmark file not found: {QUESTIONS}"
        )

    questions = pd.read_csv(QUESTIONS)

    required_columns = {
        "id",
        "type",
        "asked_at",
        "reveal_position",
        "question",
        "gold_answer",
    }

    missing = required_columns - set(questions.columns)

    if missing:
        raise ValueError(
            f"Benchmark is missing columns: {sorted(missing)}"
        )

    print("=" * 72)
    print("ANSWER SAFETY BENCHMARK PLAN")
    print("=" * 72)
    print(
        "This script classifies expected behavior. "
        "It does not call the LLM or claim to measure accuracy."
    )

    for _, row in questions.iterrows():
        behavior = expected_behavior(row)

        print(f"\n{row['id']} [{row['type']}]")
        print(f"Question: {row['question']}")
        print(f"Viewer position: {int(row['asked_at'])}")
        print(f"Reveal position: {int(row['reveal_position'])}")
        print(f"Expected behavior: {behavior}")

        if behavior == "PROTECT_FUTURE":
            print(
                "Rule: do not disclose the future outcome. "
                "Answer only if the question can be safely "
                "answered from current evidence; otherwise refuse."
            )
        elif behavior == "ANSWER":
            print(
                "Rule: answer using sufficient available evidence "
                "and cite the supporting chunks."
            )
        elif behavior == "ANSWER_IF_SUPPORTED":
            print(
                "Rule: answer only if the available evidence "
                "supports the requested outcome."
            )
        elif behavior == "OUTSIDE_CURRENT_COLLECTION":
            print(
                "Rule: do not use later-season or outside knowledge "
                "as if it were present in the current collection."
            )
        elif behavior == "UNRESOLVED_MYSTERY":
            print(
                "Rule: do not invent or assert an identity "
                "that the available evidence does not establish."
            )
        else:
            print(
                "Rule: inspect this benchmark label before "
                "using it for automated scoring."
            )

    print("\n" + "=" * 72)
    print(f"Questions reviewed: {len(questions)}")
    print("No model calls were made.")
    print("Classification review complete.")


if __name__ == "__main__":
    main()
