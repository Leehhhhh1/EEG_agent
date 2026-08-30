"""Pure ranking helpers for the multi-stage RAG pipeline."""

from typing import Any, Sequence


def reciprocal_rank_fusion(
    dense_results: list[dict[str, Any]],
    sparse_results: list[dict[str, Any]],
    *,
    dense_weight: float = 0.6,
    sparse_weight: float = 0.4,
    rank_constant: int = 60,
    top_k: int = 40,
) -> list[dict[str, Any]]:
    """Fuse dense/sparse ranks without assuming their raw scores share a scale."""
    if top_k <= 0:
        return []
    fused: dict[str, dict[str, Any]] = {}
    maximum = (dense_weight + sparse_weight) / (rank_constant + 1)
    for result_name, results, weight in (
        ("dense", dense_results, dense_weight),
        ("sparse", sparse_results, sparse_weight),
    ):
        for rank, result in enumerate(results, 1):
            chunk_id = str(result["chunk_id"])
            item = fused.setdefault(chunk_id, dict(result))
            item.update(result)
            item["hybrid_score"] = float(item.get("hybrid_score", 0.0)) + weight / (rank_constant + rank)
            item[f"{result_name}_rank"] = rank
    for item in fused.values():
        hybrid = item["hybrid_score"] / maximum if maximum else 0.0
        item["hybrid_score"] = max(0.0, min(1.0, hybrid))
        item["coarse_score"] = 2.0 * item["hybrid_score"] - 1.0
    return sorted(fused.values(), key=lambda item: item["hybrid_score"], reverse=True)[:top_k]


def fuse_three_way_ranks(
    candidates: list[dict[str, Any]],
    colbert_scores: Sequence[float],
    *,
    dense_weight: float = 0.35,
    sparse_weight: float = 0.20,
    colbert_weight: float = 0.45,
    rank_constant: int = 60,
    top_k: int = 15,
) -> list[dict[str, Any]]:
    """Fuse Dense, Sparse, and ColBERT ranks into one normalized score."""
    if len(candidates) != len(colbert_scores):
        raise ValueError("Each recalled candidate must have one ColBERT score.")
    if top_k <= 0:
        return []
    colbert_order = sorted(
        range(len(candidates)),
        key=lambda index: float(colbert_scores[index]),
        reverse=True,
    )
    colbert_ranks = {candidate_index: rank for rank, candidate_index in enumerate(colbert_order, 1)}
    maximum = (dense_weight + sparse_weight + colbert_weight) / (rank_constant + 1)
    ranked: list[dict[str, Any]] = []
    for candidate_index, candidate in enumerate(candidates):
        item = dict(candidate)
        item["colbert_score"] = float(colbert_scores[candidate_index])
        item["colbert_rank"] = colbert_ranks[candidate_index]
        score = colbert_weight / (rank_constant + item["colbert_rank"])
        if "dense_rank" in item:
            score += dense_weight / (rank_constant + int(item["dense_rank"]))
        if "sparse_rank" in item:
            score += sparse_weight / (rank_constant + int(item["sparse_rank"]))
        three_way = score / maximum if maximum else 0.0
        item["three_way_score"] = max(0.0, min(1.0, three_way))
        item["coarse_score"] = 2.0 * item["three_way_score"] - 1.0
        ranked.append(item)
    ranked.sort(key=lambda item: item["three_way_score"], reverse=True)
    return ranked[:top_k]


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
