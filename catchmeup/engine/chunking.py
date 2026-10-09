import re
from dataclasses import dataclass

@dataclass
class Chunk:
    chunk_id: str
    unit_id: int
    position: int
    label: str
    text: str

def chunk_unit(unit, max_words=150):
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", unit["text"]) if p.strip()]
    pieces, current = [], []
    for paragraph in paragraphs:
        words_now = sum(len(p.split()) for p in current)
        if current and words_now + len(paragraph.split()) > max_words:
            pieces.append("\n\n".join(current))
            current = []
        current.append(paragraph)
    if current:
        pieces.append("\n\n".join(current))
    return [
        Chunk(f"{unit['unit_id']}-c{i:02d}", unit["unit_id"],
              unit["position"], unit["label"], text)
        for i, text in enumerate(pieces, start=1)
    ]

def chunk_all(units, max_words=150):
    chunks = []
    for unit in units:
        chunks.extend(chunk_unit(unit, max_words))
    return chunks