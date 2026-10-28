
import json
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
REGISTRY_PATH = DATA_DIR / "collections.json"


@dataclass(frozen=True)
class CollectionConfig:
    collection_id: str
    name: str
    unit_name: str
    data_path: Path


COLLECTIONS = {
    "gossip-girl-s1": CollectionConfig(
        collection_id="gossip-girl-s1",
        name="Gossip Girl Season 1",
        unit_name="episode",
        data_path=DATA_DIR / "gossip-girl_s1.json",
    ),
}


def save_collection(config: CollectionConfig) -> None:
    """Persist collection metadata without storing absolute file paths."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    registry = {}
    if REGISTRY_PATH.is_file():
        with REGISTRY_PATH.open("r", encoding="utf-8") as file:
            registry = json.load(file)

    registry[config.collection_id] = {
        "name": config.name,
        "unit_name": config.unit_name,
        "data_filename": config.data_path.name,
    }

    temporary_path = REGISTRY_PATH.with_suffix(".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(registry, file, ensure_ascii=False, indent=2)

    temporary_path.replace(REGISTRY_PATH)


def get_collection(collection_id: str) -> CollectionConfig:
    """Return a registered collection or recover it from saved metadata."""
    config = COLLECTIONS.get(collection_id)

    if config is not None:
        if not config.data_path.is_file():
            raise FileNotFoundError(
                f"Dataset for collection '{collection_id}' was not found."
            )
        return config

    if not collection_id or Path(collection_id).name != collection_id:
        raise KeyError(f"Unknown collection: {collection_id}")

    registry = {}
    if REGISTRY_PATH.is_file():
        with REGISTRY_PATH.open("r", encoding="utf-8") as file:
            registry = json.load(file)

    metadata = registry.get(collection_id)

    if metadata is not None:
        filename = metadata.get("data_filename", "")
        if not filename or Path(filename).name != filename:
            raise KeyError(f"Invalid metadata for collection: {collection_id}")

        data_path = DATA_DIR / filename
        config = CollectionConfig(
            collection_id=collection_id,
            name=metadata["name"],
            unit_name=metadata["unit_name"],
            data_path=data_path,
        )
    else:
        # Backward compatibility for datasets created before the registry.
        data_path = DATA_DIR / f"{collection_id}.json"
        if not data_path.is_file():
            raise KeyError(f"Unknown collection: {collection_id}")

        config = CollectionConfig(
            collection_id=collection_id,
            name=collection_id.replace("-", " ").title(),
            unit_name="episode",
            data_path=data_path,
        )

    if not config.data_path.is_file():
        raise FileNotFoundError(
            f"Dataset for collection '{collection_id}' was not found."
        )

    COLLECTIONS[collection_id] = config
    return config
