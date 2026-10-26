
from catchmeup.content_parser import parse_episode_list

sample = """
Episodes
Season 1
1. Pilot
2. Cat's in the Bag...
3. ...And the Bag's in the River
4. Cancer Man

Season 2
1. Seven Thirty-Seven
2. Grilled
"""

units = parse_episode_list(
    text=sample,
    work_id="breaking-bad",
)

print(f"Parsed units: {len(units)}")

for unit in units:
    print(
        unit["position"],
        unit["label"],
        unit["text"],
    )
