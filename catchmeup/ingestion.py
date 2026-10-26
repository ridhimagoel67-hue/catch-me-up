
import json
import re

from catchmeup.collections import COLLECTIONS, CollectionConfig, DATA_DIR
from catchmeup.content_parser import parse_episode_table


def ingest_episode_collection(
    content: str,
    collection_id: str,
    name: str,
    source_url: str,
) -> dict:
    """Parse, validate, persist, and register an episode collection."""

    if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", collection_id):
        raise ValueError(
            "collection_id must contain lowercase letters, numbers, and hyphens."
        )

    if collection_id in COLLECTIONS:
        raise ValueError("That collection ID is already registered.")

    if not name.strip():
        raise ValueError("Collection name cannot be empty.")

    if not content.strip():
        raise ValueError("Extracted content cannot be empty.")

    units = parse_episode_table(content, work_id=collection_id)

    if not units:
        raise ValueError("No episode table could be parsed from this content.")

    positions = [unit["position"] for unit in units]
    unit_ids = [unit["unit_id"] for unit in units]

    if positions != list(range(1, len(units) + 1)):
        raise ValueError("Episode positions must be sequential starting at 1.")

    if len(unit_ids) != len(set(unit_ids)):
        raise ValueError("Duplicate episode IDs were found.")

    # Include source metadata without changing the existing loader's required fields.
    for unit in units:
        unit["source_url"] = source_url

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    data_path = DATA_DIR / f"{collection_id}.json"

    if data_path.exists():
        raise ValueError("A dataset file with that collection ID already exists.")

    # Persist the complete dataset before exposing it in the registry.
    with data_path.open("x", encoding="utf-8") as file:
        json.dump(units, file, ensure_ascii=False, indent=2)

    config = CollectionConfig(
        collection_id=collection_id,
        name=name.strip(),
        unit_name="episode",
        data_path=data_path,
    )

    COLLECTIONS[collection_id] = config

    return {
        "collection_id": collection_id,
        "name": name.strip(),
        "unit_name": "episode",
        "unit_count": len(units),
        "first_position": positions[0],
        "last_position": positions[-1],
        "data_path": str(data_path),
    }
