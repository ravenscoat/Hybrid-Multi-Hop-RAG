from __future__ import annotations

import math
import re
from collections import Counter


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold(), flags=re.UNICODE)


class BM25:
    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.term_freqs = [Counter(tokenize(document)) for document in documents]
        self.lengths = [sum(freq.values()) for freq in self.term_freqs]
        self.avg_length = sum(self.lengths) / max(1, len(self.lengths))
        doc_freq: Counter[str] = Counter()
        for freq in self.term_freqs:
            doc_freq.update(freq.keys())
        count = len(documents)
        self.idf = {
            term: math.log(1 + (count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in doc_freq.items()
        }

    def search(self, query: str, limit: int) -> list[tuple[int, float]]:
        terms = tokenize(query)
        scores: list[tuple[int, float]] = []
        for doc_id, frequencies in enumerate(self.term_freqs):
            length_norm = 1 - self.b + self.b * self.lengths[doc_id] / max(self.avg_length, 1)
            score = 0.0
            for term in terms:
                frequency = frequencies.get(term, 0)
                if frequency:
                    score += self.idf.get(term, 0.0) * (
                        frequency * (self.k1 + 1) / (frequency + self.k1 * length_norm)
                    )
            if score > 0:
                scores.append((doc_id, score))
        return sorted(scores, key=lambda item: item[1], reverse=True)[:limit]

