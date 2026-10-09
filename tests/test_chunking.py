from catchmeup.engine.chunking import chunk_unit
def make_unit():
    para = " ".join(["word"] * 60)
    return {"unit_id": "s01e01", "position": 1, "label": "S1E1",
            "text": "\n\n".join([para] * 4)}


def test_paragraphs_are_grouped_up_to_max_words():
    chunks = chunk_unit(make_unit(), max_words=150)
    assert len(chunks) == 2
    assert [c.chunk_id for c in chunks] == ["s01e01-c01", "s01e01-c02"]


def test_chunks_keep_their_episode_position():
    assert all(c.position == 1 for c in chunk_unit(make_unit()))