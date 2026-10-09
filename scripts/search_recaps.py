import json
import re
import sys
from pathlib import Path

DATA = Path("data/processed/gossip-girl_s1.json")

keyword = " ".join(sys.argv[1:])
units = json.loads(DATA.read_text(encoding="utf-8"))
pattern = re.compile(re.escape(keyword), re.IGNORECASE)

for u in units:
    sentences = re.split(r"(?<=[.!?])\s+", u["text"])
    hits = [s for s in sentences if pattern.search(s)]
    if hits:
        print(f"\n{u['label']} - {u['meta']['title']}  ({len(hits)} hit(s))")
        for s in hits[:2]:
            print("   ", s[:220])