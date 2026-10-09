import re
from dataclasses import dataclass


@dataclass
class Chunk:
    chunk_id: str
    unit_id: str
    position: int
    label: str
    text: str


def split_long_paragraph(paragraph, max_words):
    
    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    pieces, current = [], []
    for sentence in sentences:
        words_now = sum(len(s.split()) for s in current)
        if current and words_now + len(sentence.split()) > max_words:
            pieces.append(" ".join(current))
            current = []
        current.append(sentence)
    if current:
        pieces.append(" ".join(current))
    return pieces


def chunk_unit(unit, max_words=110):
    
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", unit["text"]) if p.strip()]

    parts = []
    for paragraph in paragraphs:
        if len(paragraph.split()) > max_words:
            parts.extend(split_long_paragraph(paragraph, max_words))
        else:
            parts.append(paragraph)

    pieces, current = [], []
    for part in parts:
        words_now = sum(len(p.split()) for p in current)
        if current and words_now + len(part.split()) > max_words:
            pieces.append("\n\n".join(current))
            current = []
        current.append(part)
    if current:
        pieces.append("\n\n".join(current))

    return [
        Chunk(f"{unit['unit_id']}-c{i:02d}", unit["unit_id"],
              unit["position"], unit["label"], text)
        for i, text in enumerate(pieces, start=1)
    ]


def chunk_all(units, max_words=110):
    chunks = []
    for unit in units:
        chunks.extend(chunk_unit(unit, max_words))
    return chunks