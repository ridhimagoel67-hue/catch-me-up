
from pathlib import Path

from catchmeup.content_parser import parse_episode_table

source = Path("data/processed/breaking_bad_extracted.txt")
text = source.read_text(encoding="utf-8")

episodes = parse_episode_table(text, work_id="breaking_bad")

print("Episodes parsed:", len(episodes))
print("First 5 episodes:")
for episode in episodes[:5]:
    print(episode)

print("\nLast 5 episodes:")
for episode in episodes[-5:]:
    print(episode)

positions = [episode["position"] for episode in episodes]
unit_ids = [episode["unit_id"] for episode in episodes]

print("\nPositions sequential:", positions == list(range(1, 63)))
print("Unique unit IDs:", len(unit_ids) == len(set(unit_ids)))
print("All 62 episodes found:", len(episodes) == 62)
