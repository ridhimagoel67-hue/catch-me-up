
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from catchmeup.engine.chunking import chunk_all


OUTPUT_FILE = Path("data/processed/generated_questions.jsonl")


def validate_question(question_data, chunk):
    required = ["question", "answer", "support_quote"]

    for field in required:
        if not isinstance(question_data.get(field), str):
            return False

        if not question_data[field].strip():
            return False

    return question_data["support_quote"] in chunk.text


def generate_question(client, model, chunk):
    prompt = f"""
Generate exactly one question-answer pair from this passage.

Rules:
- Ask a clear, self-contained question.
- Answer using only this passage.
- Copy one supporting quote exactly from the passage.
- Do not invent facts or use outside knowledge.
- Return only valid JSON with these keys:
  question, answer, support_quote.

Passage:
{chunk.text}
"""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000,
        temperature=0,
    )

    raw = response.choices[0].message.content

    if not raw or not raw.strip():
        raise ValueError("Model returned empty text")

    question_data = json.loads(raw)

    if not isinstance(question_data, dict):
        raise ValueError("Response must be a JSON object")

    if not validate_question(question_data, chunk):
        raise ValueError("Invalid fields or supporting quote")

    return {
        "chunk_id": chunk.chunk_id,
        "unit_id": chunk.unit_id,
        "reveal_position": chunk.position,
        "question": question_data["question"],
        "answer": question_data["answer"],
        "support_quote": question_data["support_quote"],
    }


def load_completed_chunks():
    """Read chunk IDs already saved, allowing safe restarts."""
    completed = set()

    if not OUTPUT_FILE.exists():
        return completed

    with OUTPUT_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            try:
                record = json.loads(line)
                completed.add(record["chunk_id"])
            except (json.JSONDecodeError, KeyError):
                # Ignore malformed lines when reading progress.
                continue

    return completed


def main():
    load_dotenv(encoding="utf-8-sig")

    client = OpenAI(
        api_key=os.environ["LLM_API_KEY"],
        base_url=os.environ["LLM_BASE_URL"],
    )
    model = os.environ["LLM_MODEL"]

    with open(
        "data/processed/gossip-girl_s1.json",
        encoding="utf-8",
    ) as file:
        units = json.load(file)

    chunks = chunk_all(units)
    print("Total chunks:", len(chunks))

    # For our first test, process only three chunks.
    test_chunks = chunks[:3]

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    completed = load_completed_chunks()

    print("Already saved:", len(completed))
    generated_this_run = 0

    for chunk in test_chunks:
        if chunk.chunk_id in completed:
            print("Skipping already saved:", chunk.chunk_id)
            continue

        print("Generating:", chunk.chunk_id)

        # Retry temporary failures up to three times.
        for attempt in range(1, 4):
            try:
                result = generate_question(client, model, chunk)

                # Append one complete JSON record per line.
                with OUTPUT_FILE.open("a", encoding="utf-8") as file:
                    file.write(
                        json.dumps(result, ensure_ascii=False) + "\n"
                    )

                completed.add(chunk.chunk_id)
                generated_this_run += 1
                print("Saved:", chunk.chunk_id)
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

        # Small pause between successful chunk attempts.
        time.sleep(2)

    print("\nFinished this run.")
    print("New questions saved:", generated_this_run)
    print("Output file:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
