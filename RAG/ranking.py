"""Pure ranking helpers for the two-stage RAG pipeline."""

from typing import Any, Sequence


def fuse_scores_and_deduplicate_sources(
    candidates: list[dict[str, Any]],
    rerank_scores: Sequence[float],
    coarse_weight: float = 0.2,
    rerank_weight: float = 0.8,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    if len(candidates) != len(rerank_scores):
        raise ValueError("Each FAISS candidate must have one reranker score.")
    if top_k <= 0:
        return []

    ranked = []
    for candidate, rerank_score in zip(candidates, rerank_scores):
        item = dict(candidate)
        coarse_score = max(0.0, min(1.0, (float(item["coarse_score"]) + 1.0) / 2.0))
        item["coarse_normalized"] = coarse_score
        item["rerank_score"] = float(rerank_score)
        item["combined_score"] = (
            coarse_weight * coarse_score
            + rerank_weight * float(rerank_score)
        )
        ranked.append(item)

    ranked.sort(key=lambda item: item["combined_score"], reverse=True)
    deduplicated = []
    seen_sources = set()
    for item in ranked:
        source_key = item["source"].casefold()
        if source_key in seen_sources:
            continue
        seen_sources.add(source_key)
        deduplicated.append(item)
        if len(deduplicated) == top_k:
            break
    return deduplicated
