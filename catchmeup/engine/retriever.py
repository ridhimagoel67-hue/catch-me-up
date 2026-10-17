
import re
from collections import Counter

import numpy as np


class Retriever:
    def __init__(self, chunks, embed_fn):
        self.chunks = chunks
        self.embed = embed_fn

        # Existing vector-search baseline.
        self.vectors = np.asarray(
            self.embed([chunk.text for chunk in chunks])
        )
        self.positions = np.array(
            [chunk.position for chunk in chunks]
        )

        # Pre-tokenize documents for BM25.
        self.tokenized_docs = [
            self._tokenize(chunk.text) for chunk in chunks
        ]
        self.doc_lengths = np.array(
            [len(tokens) for tokens in self.tokenized_docs]
        )

    @staticmethod
    def _tokenize(text):
        """Lowercase text and extract word tokens."""
        return re.findall(r"\b\w+\b", text.lower())

    def _allowed_indices(self, position):
        """Return only chunks available at the requested position."""
        return np.where(self.positions <= position)[0]

    def _vector_ranked(self, query, allowed):
        """Rank allowed chunks using the existing vector scores."""
        if len(allowed) == 0:
            return []

        query_vector = np.asarray(self.embed([query]))[0]
        scores = self.vectors[allowed] @ query_vector
        order = np.argsort(-scores)

        return [
            (int(allowed[i]), float(scores[i]))
            for i in order
        ]

    def _bm25_ranked(self, query, allowed, k1=1.5, b=0.75):
        """
        Rank allowed chunks using BM25 keyword relevance.

        Statistics are computed only from allowed chunks so future
        documents do not affect the current retrieval ranking.
        """
        if len(allowed) == 0:
            return []

        query_tokens = self._tokenize(query)

        if not query_tokens:
            return []

        # Calculate document frequencies within allowed content only.
        allowed_docs = [
            self.tokenized_docs[i] for i in allowed
        ]

        doc_count = len(allowed_docs)
        doc_freq = Counter()

        for tokens in allowed_docs:
            doc_freq.update(set(tokens))

        average_length = (
            sum(len(tokens) for tokens in allowed_docs) / doc_count
        )

        if average_length == 0:
            return []

        scores = []

        for index, tokens in zip(allowed, allowed_docs):
            term_counts = Counter(tokens)
            doc_length = len(tokens)
            score = 0.0

            for term in query_tokens:
                frequency = term_counts.get(term, 0)

                if frequency == 0:
                    continue

                df = doc_freq[term]

                # BM25 inverse document frequency.
                idf = np.log(
                    1 + (doc_count - df + 0.5) / (df + 0.5)
                )

                denominator = frequency + k1 * (
                    1 - b + b * doc_length / average_length
                )

                score += idf * (
                    frequency * (k1 + 1) / denominator
                )

            scores.append((int(index), float(score)))

        scores.sort(key=lambda item: item[1], reverse=True)
        return scores

    def search(self, query, position, k=5):
        """
        Original vector-search baseline.

        Kept as the default so existing code and tests remain compatible.
        """
        allowed = self._allowed_indices(position)
        ranked = self._vector_ranked(query, allowed)

        return [
            (self.chunks[index], score)
            for index, score in ranked[:k]
        ]

    def search_bm25(self, query, position, k=5):
        """Search using keyword-based BM25 ranking."""
        allowed = self._allowed_indices(position)
        ranked = self._bm25_ranked(query, allowed)

        return [
            (self.chunks[index], score)
            for index, score in ranked[:k]
        ]

    def search_hybrid(self, query, position, k=5, rrf_k=60):
        """
        Combine vector and BM25 rankings using Reciprocal Rank Fusion.

        The returned scores are fusion scores, not cosine similarities
        or BM25 scores.
        """
        allowed = self._allowed_indices(position)

        if len(allowed) == 0:
            return []

        vector_results = self._vector_ranked(query, allowed)
        bm25_results = self._bm25_ranked(query, allowed)

        fused_scores = {}

        # Rank positions start at 1.
        for ranked_results in (vector_results, bm25_results):
            for rank, (index, score) in enumerate(
                ranked_results, start=1
            ):
                fused_scores[index] = (
                    fused_scores.get(index, 0.0)
                    + 1.0 / (rrf_k + rank)
                )

        ranked_indices = sorted(
            fused_scores,
            key=lambda index: fused_scores[index],
            reverse=True,
        )

        return [
            (self.chunks[index], fused_scores[index])
            for index in ranked_indices[:k]
        ]
