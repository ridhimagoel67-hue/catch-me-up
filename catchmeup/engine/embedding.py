from sentence_transformers import SentenceTransformer

_MODEL=None 
def embed(texts):
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _MODEL.encode(texts,normalize_embeddings=True,show_progress_bar=False)