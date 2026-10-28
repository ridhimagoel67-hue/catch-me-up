"""Validated persistence for universal ordered-content collections."""

import json
import re
from pathlib import Path
from urllib.parse import urlparse

from catchmeup.collections import COLLECTIONS, CollectionConfig, DATA_DIR, save_collection
from catchmeup.content_parser import parse_ordered_content

CONTENT_TYPES = {
    "show": "episode",
    "book": "chapter",
    "course": "lesson",
    "podcast": "episode",
    "other": "section",
}


def ingest_content_collection(
    content: str,
    collection_id: str,
    name: str,
    source_url: str,
    content_type: str = "show",
) -> dict:
    """Parse, validate, persist and register an ordered content collection."""
    if not isinstance(collection_id, str) or not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", collection_id):
        raise ValueError("collection_id must contain lowercase letters, numbers, and hyphens.")
    if collection_id in COLLECTIONS:
        raise ValueError("That collection ID is already registered.")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Collection name cannot be empty.")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Extracted content cannot be empty.")
    if content_type not in CONTENT_TYPES:
        raise ValueError("content_type must be show, book, course, podcast, or other.")
    parsed_url = urlparse(source_url.strip()) if isinstance(source_url, str) else None
    if not parsed_url or parsed_url.scheme != "https" or not parsed_url.hostname or parsed_url.username or parsed_url.password:
        raise ValueError("source_url must be a valid HTTPS URL.")

    units = parse_ordered_content(content, work_id=collection_id, content_type=content_type)
    if not units:
        expected = {
            "show": "an episode list/table",
            "book": "ordered chapter headings, a numbered chapter list, or a chapter table",
            "course": "ordered lesson/module headings, a numbered lesson list, or a course table",
            "podcast": "ordered episode headings or a numbered episode list",
            "other": "ordered section headings or a numbered list",
        }[content_type]
        raise ValueError(f"Could not identify {expected} in the extracted source. Try a different structured source.")

    # Require sequential positions and unique IDs before writing anything.
    positions = [unit.get("position") for unit in units]
    unit_ids = [unit.get("unit_id") for unit in units]
    if positions != list(range(1, len(units) + 1)):
        raise ValueError("Content positions must be sequential starting at 1.")
    if len(unit_ids) != len(set(unit_ids)):
        raise ValueError("Duplicate content unit IDs were found.")

    for unit in units:
        unit["source_url"] = source_url.strip()
        unit["source_type"] = content_type

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    data_path = DATA_DIR / f"{collection_id}.json"
    if data_path.exists():
        raise ValueError("A dataset file with that collection ID already exists.")

    # Write dataset first; register it only after the dataset is complete.
    with data_path.open("x", encoding="utf-8") as file:
        json.dump(units, file, ensure_ascii=False, indent=2)

    unit_name = CONTENT_TYPES[content_type]
    config = CollectionConfig(
        collection_id=collection_id,
        name=name.strip(),
        unit_name=unit_name,
        data_path=data_path,
    )
    try:
        save_collection(config)
    except Exception:
        # Avoid leaving an orphan dataset if registry persistence fails.
        try:
            data_path.unlink(missing_ok=True)
        finally:
            raise
    COLLECTIONS[collection_id] = config

    return {
        "collection_id": collection_id,
        "name": name.strip(),
        "content_type": content_type,
        "unit_name": unit_name,
        "unit_count": len(units),
        "first_position": positions[0],
        "last_position": positions[-1],
        "data_path": str(data_path),
    }


def ingest_episode_collection(content: str, collection_id: str, name: str, source_url: str) -> dict:
    """Backward-compatible wrapper for existing callers and tests."""
    return ingest_content_collection(
        content=content,
        collection_id=collection_id,
        name=name,
        source_url=source_url,
        content_type="show",
    )
