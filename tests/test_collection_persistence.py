
import json

from catchmeup.collections import (
    COLLECTIONS,
    REGISTRY_PATH,
    CollectionConfig,
    get_collection,
    save_collection,
)


def test_custom_collection_metadata_survives_registry_reload(tmp_path, monkeypatch):
    data_dir = tmp_path / "processed"
    data_dir.mkdir()
    registry_path = data_dir / "collections.json"
    dataset_path = data_dir / "my-book.json"
    dataset_path.write_text("[]", encoding="utf-8")

    monkeypatch.setattr("catchmeup.collections.DATA_DIR", data_dir)
    monkeypatch.setattr("catchmeup.collections.REGISTRY_PATH", registry_path)

    config = CollectionConfig(
        collection_id="my-book",
        name="My Original Book Name",
        unit_name="chapter",
        data_path=dataset_path,
    )
    save_collection(config)

    # Simulate a fresh process with an empty in-memory registry.
    COLLECTIONS.pop("my-book", None)

    restored = get_collection("my-book")

    assert restored.name == "My Original Book Name"
    assert restored.unit_name == "chapter"
    assert restored.data_path == dataset_path

    saved = json.loads(registry_path.read_text(encoding="utf-8"))
    assert saved["my-book"]["data_filename"] == "my-book.json"
