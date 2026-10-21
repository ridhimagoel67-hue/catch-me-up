
from pathlib import Path

import pytest

from catchmeup.collections import (
    COLLECTIONS,
    CollectionConfig,
    get_collection,
)


def test_gossip_girl_collection_is_registered():
    config = get_collection("gossip-girl-s1")

    assert config.name == "Gossip Girl Season 1"
    assert config.unit_name == "episode"
    assert config.data_path.is_file()


def test_unknown_collection_is_rejected():
    with pytest.raises(KeyError, match="Unknown collection"):
        get_collection("not-registered")


def test_missing_dataset_is_rejected(tmp_path):
    COLLECTIONS["missing-test"] = CollectionConfig(
        collection_id="missing-test",
        name="Missing test",
        unit_name="chapter",
        data_path=tmp_path / "missing.json",
    )

    try:
        with pytest.raises(FileNotFoundError):
            get_collection("missing-test")
    finally:
        del COLLECTIONS["missing-test"]
