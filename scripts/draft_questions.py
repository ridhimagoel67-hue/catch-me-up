import csv
import json
import random
import re
from pathlib import Path

DATA = Path("data/processed/gossip-girl_s1.json")
OUT = Path("data/benchmark/questions_draft.csv")

KEYWORDS = re.compile(
    r"\b(kiss\w*|dating|boyfriend|girlfriend|break\w* up|broke up|reveal\w*|"
    r"admit\w*|confess\w*|arrest\w*|expel\w*|fire[sd]?|sleep\w*|slept|"
    r"engag\w*|secret\w*|caught|blackmail\w*|reject\w*|accept\w*)\b",
    re.IGNORECASE,
)
BAD_START = re.compile(
    r"^(he|she|they|it|his|her|their|this|that|then|meanwhile|however|afterward|later)\b",
    re.IGNORECASE,
)
FIELDS = ["id", "type", "asked_at", "reveal_position", "evidence_unit",
          "question", "gold_answer", "evidence"]


def candidates(unit):
    """Return self-contained, fact-like sentences, best first."""
    sentences = re.split(r"(?<=[.!?])\s+", unit["text"])
    good = []
    for s in sentences:
        s = s.strip()
        if not 60 <= len(s) <= 220:
            continue
        if BAD_START.match(s):
            continue
        hits = len(KEYWORDS.findall(s))
        if hits:
            good.append((hits, s))
    good.sort(key=lambda pair: -pair[0])
    return [s for _, s in good]


def main():
    units = json.loads(DATA.read_text(encoding="utf-8"))
    by_pos = {u["position"]: u for u in units}
    rng = random.Random(7)
    rows = []

    def add(kind, episode, asked_at):
        if episode not in by_pos:
            return
        sentences = candidates(by_pos[episode])
        if not sentences:
            return
        rows.append({
            "id": f"{kind}{len(rows) + 1}",
            "type": kind,
            "asked_at": asked_at,
            "reveal_position": episode,
            "evidence_unit": by_pos[episode]["unit_id"],
            "question": "",
            "gold_answer": "",
            "evidence": rng.choice(sentences[:3]),
        })

    for ep in [3, 4, 5, 6, 7]:
        add("A", ep, min(ep + 2, 18))
    for ep in [9, 10, 11, 13, 15, 16, 17]:
        add("B", ep, ep - 4)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    for r in rows:
        print(f"{r['id']} | type {r['type']} | asked at {r['asked_at']} | "
              f"answer in {r['evidence_unit']}\n    {r['evidence']}\n")
    print(f"Saved {len(rows)} candidate rows to {OUT}")


if __name__ == "__main__":
    main()