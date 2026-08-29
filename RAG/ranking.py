"""Pure ranking helpers for the two-stage RAG pipeline."""

from typing import Any, Sequence


def passes_faiss_probe(
    candidates: list[dict[str, Any]],
    threshold: float,
) -> bool:
    """Return whether the best FAISS candidate is relevant enough to rerank."""
    return bool(candidates) and float(candidates[0]["coarse_score"]) >= threshold


def filter_by_rerank_threshold(
    ranked: list[dict[str, Any]],
    threshold: float,
    top_k: int,
) -> list[dict[str, Any]]:
    """Produce dynamic Top-0..K output after reranker relevance filtering."""
    if top_k <= 0:
        return []
    return [
        item for item in ranked
        if float(item["rerank_score"]) >= threshold
    ][:top_k]


def fuse_scores_and_select_top_k(
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
    return ranked[:top_k]
