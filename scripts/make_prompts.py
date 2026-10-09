from pathlib import Path

import pandas as pd

df = pd.read_csv("data/benchmark/questions_gg_s1.csv")

lines = []
for _, r in df.iterrows():
    lines.append(f"===== {r['id']} ")
    lines.append(
        f"I'm watching Gossip Girl. I have finished episode {r['asked_at']} "
        f"of season 1 and have not seen anything after it. "
        f"Please answer without spoilers: {r['question']}"
    )
    lines.append("")
Path("data/benchmark/prompts.txt").write_text("\n".join(lines), encoding="utf-8")

results = pd.DataFrame({
    "id": df["id"], "type": df["type"],
    "leaked": "", "correct": "", "refused": "", "note": "",
})
results.to_csv("data/benchmark/chatbot_results.csv", index=False)
print("Wrote data/benchmark/prompts.txt and data/benchmark/chatbot_results.csv")