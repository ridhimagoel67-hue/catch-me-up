import zlib

import numpy as np

from catchmeup.engine.chunking import Chunk
from catchmeup.engine.retriever import Retriever


def fake_embed(texts):
    
    vectors = []
    for t in texts:
        rng = np.random.default_rng(zlib.crc32(t.encode("utf-8")))
        v = rng.normal(size=16)
        vectors.append(v / np.linalg.norm(v))
    return np.array(vectors)


def make_chunks():
    return [Chunk(f"s01e{p:02d}-c{i}", f"s01e{p:02d}", p, f"S1E{p}", f"text {p} {i}")
            for p in range(1, 11) for i in (1, 2, 3)]


def test_never_returns_chunks_after_the_viewers_position():
    retriever = Retriever(make_chunks(), fake_embed)
    for position in range(0, 11):
        for query in ["who kissed whom", "wedding", "secret", "college"]:
            for chunk, _ in retriever.search(query, position, k=50):
                assert chunk.position <= position


def test_position_zero_returns_nothing():
    assert Retriever(make_chunks(), fake_embed).search("anything", 0) == []


def test_returns_k_results_when_enough_are_visible():
    retriever = Retriever(make_chunks(), fake_embed)
    assert len(retriever.search("anything", 5, k=4)) == 4