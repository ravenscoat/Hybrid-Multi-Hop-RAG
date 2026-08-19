from __future__ import annotations

from collections import defaultdict


def reciprocal_rank_fusion(
    rankings: list[list[int]],
    constant: int = 60,
    limit: int | None = None,
    weights: list[float] | None = None,
) -> list[tuple[int, float]]:
    """Fuse ranked document IDs, optionally weighting each ranking."""
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights must have one value per ranking")
    scores: dict[int, float] = defaultdict(float)
    for ranking, weight in zip(rankings, weights):
        seen: set[int] = set()
        for rank, doc_id in enumerate(ranking, start=1):
            if doc_id not in seen:
                scores[doc_id] += float(weight) / (constant + rank)
                seen.add(doc_id)
    result = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return result[:limit] if limit is not None else result
