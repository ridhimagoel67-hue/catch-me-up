
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from catchmeup.engine.chunking import chunk_all


OUTPUT_FILE = Path("data/processed/generated_questions.jsonl")


def validate_question(question_data, chunk):
    """Validate the structure and evidence of a generated record."""

    if not isinstance(question_data, dict):
        return False

    # The model must explicitly classify suitability.
    if not isinstance(question_data.get("suitable"), bool):
        return False

    # A rejected chunk is valid if it includes a reason
    # and empty question, answer, and supporting quote fields.
    if question_data["suitable"] is False:
        return (
            question_data.get("question") == ""
            and question_data.get("answer") == ""
            and question_data.get("support_quote") == ""
            and isinstance(question_data.get("reason"), str)
            and bool(question_data["reason"].strip())
        )

    # Suitable questions must contain all required fields.
    required = ["question", "answer", "support_quote"]

    for field in required:
        value = question_data.get(field)

        if not isinstance(value, str) or not value.strip():
            return False

    # The supporting quote must occur in the source passage.
    return question_data["support_quote"] in chunk.text


def generate_question(client, model, chunk):
    """Generate one question or record why a chunk is unsuitable."""

    prompt = f"""
You are generating a high-quality evaluation question for a
spoiler-safe question-answering system.

Read the passage carefully.

Return exactly one JSON object with these keys:
suitable, question, answer, support_quote, reason.

Rules:
- suitable must be true or false.
- Set suitable to false if the passage cannot support a clear,
  factual, self-contained question.
- Set suitable to false if the passage is too vague or lacks
  enough information for a reliable answer.
- Prefer questions about concrete events, people, actions,
  locations, or explicitly stated facts.
- Avoid vague questions, subjective interpretations, and
  questions requiring outside knowledge.
- The answer must be supported by the passage alone.
- Copy support_quote exactly from the passage.
- If suitable is false, use empty strings for question,
  answer, and support_quote, and explain why in reason.
- If suitable is true, provide a non-empty question, answer,
  and exact supporting quote.
- Return valid JSON only. Do not include Markdown fences.

Passage:
{chunk.text}
"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=1000,
        temperature=0,
    )

    raw = response.choices[0].message.content

    if not raw or not raw.strip():
        raise ValueError("Model returned empty text")

    question_data = json.loads(raw)

    if not validate_question(question_data, chunk):
        print("\nDEBUG: Invalid model response")
        print("Chunk:", chunk.chunk_id)
        print(json.dumps(
            question_data,
            indent=2,
            ensure_ascii=False,
        ))
        raise ValueError(
            f"Validation failed for {chunk.chunk_id}"
        )

    # Record source metadata from our own chunk, not the LLM.
    if question_data["suitable"] is False:
        return {
            "chunk_id": chunk.chunk_id,
            "unit_id": chunk.unit_id,
            "reveal_position": chunk.position,
            "suitable": False,
            "reason": question_data["reason"],
        }

    return {
        "chunk_id": chunk.chunk_id,
        "unit_id": chunk.unit_id,
        "reveal_position": chunk.position,
        "suitable": True,
        "question": question_data["question"],
        "answer": question_data["answer"],
        "support_quote": question_data["support_quote"],
        "reason": question_data["reason"],
    }


def load_completed_chunks():
    """Load chunk IDs already saved so we can resume safely."""

    completed = set()

    if not OUTPUT_FILE.exists():
        return completed

    with OUTPUT_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            try:
                record = json.loads(line)
                completed.add(record["chunk_id"])
            except (json.JSONDecodeError, KeyError):
                # Ignore malformed records when reading progress.
                continue

    return completed


def main():
    # 1. Load configuration.
    load_dotenv(encoding="utf-8-sig")

    client = OpenAI(
        api_key=os.environ["LLM_API_KEY"],
        base_url=os.environ["LLM_BASE_URL"],
    )
    model = os.environ["LLM_MODEL"]

    # 2. Load the episode data.
    with open(
        "data/processed/gossip-girl_s1.json",
        encoding="utf-8",
    ) as file:
        units = json.load(file)

    # 3. Split episodes into chunks.
    chunks = chunk_all(units)
    print("Total chunks:", len(chunks))

    # 4. Select up to five chunks from episode 2 onward.
    completed = load_completed_chunks()

    test_chunks = [
        chunk for chunk in chunks
        if chunk.position >= 2
        and chunk.chunk_id not in completed
    ][:20]
    # 5. Prepare the output and resume state.
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    completed = load_completed_chunks()

    print("Already saved:", len(completed))
    generated_this_run = 0

    # 6. Generate, validate, and save each record.
    for chunk in test_chunks:
        if chunk.chunk_id in completed:
            print("Skipping already saved:", chunk.chunk_id)
            continue

        print("Generating:", chunk.chunk_id)

        for attempt in range(1, 4):
            try:
                result = generate_question(client, model, chunk)

                # Append one JSON record per line.
                with OUTPUT_FILE.open(
                    "a",
                    encoding="utf-8",
                ) as file:
                    file.write(
                        json.dumps(result, ensure_ascii=False)
                        + "\n"
                    )

                completed.add(chunk.chunk_id)
                generated_this_run += 1

                if result["suitable"]:
                    print("Saved suitable question:", chunk.chunk_id)
                else:
                    print(
                        "Saved rejected chunk:",
                        chunk.chunk_id,
                        "| Reason:",
                        result["reason"],
                    )

                break

            except Exception as error:
                print(
                    f"Attempt {attempt}/3 failed:",
                    type(error).__name__,
                    str(error)[:200],
                )

                if attempt < 3:
                    time.sleep(2 ** attempt)
                else:
                    print("Giving up on this chunk for now.")

        # Brief pause between chunks.
        time.sleep(2)

    print("\nFinished this run.")
    print("New records saved:", generated_this_run)
    print("Output file:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
