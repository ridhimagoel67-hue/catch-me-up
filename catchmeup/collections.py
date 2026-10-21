
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"


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


def get_collection(collection_id: str) -> CollectionConfig:
    """Return a registered collection or reject an unknown identifier."""

    config = COLLECTIONS.get(collection_id)

    if config is None:
        raise KeyError(f"Unknown collection: {collection_id}")

    if not config.data_path.is_file():
        raise FileNotFoundError(
            f"Dataset for collection '{collection_id}' was not found."
        )

    return config
