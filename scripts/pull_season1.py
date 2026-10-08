import json
from dataclasses import asdict
from pathlib import Path

from catchmeup.adapters.fandom import fetch_recap
from catchmeup.schema import Unit

WIKI = "gossipgirl"
WORK_ID = "gossip-girl"
EPISODE_LIST = Path("data/raw/episodes_s1.txt")
OUT_FILE = Path("data/processed/gossip-girl_s1.json")


def main():
    lines = EPISODE_LIST.read_text(encoding="utf-8-sig").splitlines()
    titles = [line.strip() for line in lines if line.strip()]
    units = []
    print(f"{'pos':>3}  {'words':>5}  title")
    for position, title in enumerate(titles, start=1):
        text = fetch_recap(WIKI, title)
        if text is None:
            print(f"{position:>3}  {'---':>5}  {title}   <-- NOT FOUND")
            continue
        units.append(Unit(
            work_id=WORK_ID,
            unit_id=f"s01e{position:02d}",
            position=position,
            label=f"S1E{position}",
            text=text,
            source_type="wiki",
            source_url=f"https://{WIKI}.fandom.com/wiki/{title.replace(' ', '_')}",
            meta={"title": title},
        ))
        print(f"{position:>3}  {len(text.split()):>5}  {title}")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(
        json.dumps([asdict(u) for u in units], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nSaved {len(units)} of {len(titles)} episodes to {OUT_FILE}")


if __name__ == "__main__":
    main()