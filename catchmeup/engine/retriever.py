import numpy as np
class Retriever:
    def __init__(self, chunks, embed_fn):
        self.chunks = chunks
        self.embed = embed_fn
        self.vectors = np.asarray(self.embed([c.text for c in chunks]))
        self.positions = np.array([c.position for c in chunks])

    def search(self, query, position, k=5):
        allowed = np.where(self.positions <= position)[0]
        if len(allowed) == 0:
            return []
        q = np.asarray(self.embed([query]))[0]
        scores = self.vectors[allowed] @ q
        order = np.argsort(-scores)[:k]
        return [(self.chunks[allowed[i]], float(scores[i])) for i in order]