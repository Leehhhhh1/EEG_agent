"""Three-way BGE-M3 child retrieval with reranking and neighbor expansion."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .embedder import BGEEmbedder
from .indexer import update_index
from .ranking import (
    filter_by_rerank_threshold,
    fuse_scores_and_select_top_k,
    passes_faiss_probe,
    reciprocal_rank_fusion,
    fuse_three_way_ranks,
)
from .reranker import BGEReranker
from .searcher import HybridSearcher


DENSE_TOP_K = 20
SPARSE_TOP_K = 20
HYBRID_TOP_K = 40
THREE_WAY_TOP_K = 15
FINAL_TOP_K = 3
COARSE_WEIGHT = 0.2
RERANK_WEIGHT = 0.8
DEFAULT_FAISS_PROBE_THRESHOLD = 0.35
DEFAULT_RERANK_THRESHOLD = 0.5


class EEGRetriever:
    def __init__(self, faiss_probe_threshold: float | None = None, rerank_threshold: float | None = None):
        self.rag_dir = Path(__file__).resolve().parent
        self.faiss_probe_threshold = (
            float(os.getenv("RAG_FAISS_PROBE_THRESHOLD", DEFAULT_FAISS_PROBE_THRESHOLD))
            if faiss_probe_threshold is None else float(faiss_probe_threshold)
        )
        self.rerank_threshold = (
            float(os.getenv("RAG_RERANK_THRESHOLD", DEFAULT_RERANK_THRESHOLD))
            if rerank_threshold is None else float(rerank_threshold)
        )
        self.embedder = BGEEmbedder()
        update_index(embedder=self.embedder)
        self.searcher = HybridSearcher(
            str(self.rag_dir / "faiss.index"),
            str(self.rag_dir / "children.pkl"),
            str(self.rag_dir / "parents.pkl"),
            str(self.rag_dir / "sparse_index.pkl"),
        )
        self.reranker = BGEReranker()

    def retrieve(self, query: str, *, require_faiss_probe: bool = False) -> list[dict[str, Any]]:
        query_output = self.embedder.encode_hybrid(
            [query], return_colbert=True, max_length=256
        )
        dense_candidates = self.searcher.search_dense(
            query_output["dense_vecs"][0], top_k=DENSE_TOP_K
        )
        if not dense_candidates:
            return []
        if require_faiss_probe:
            probe_candidates = [{"coarse_score": item["dense_score"]} for item in dense_candidates]
            if not passes_faiss_probe(probe_candidates, self.faiss_probe_threshold):
                return []

        sparse_candidates = self.searcher.search_sparse(
            query_output["lexical_weights"][0], top_k=SPARSE_TOP_K
        )
        candidates = reciprocal_rank_fusion(
            dense_candidates, sparse_candidates, top_k=HYBRID_TOP_K
        )
        colbert_scores = self.embedder.score_colbert(
            query_output["colbert_vecs"][0],
            [candidate["retrieval_text"] for candidate in candidates],
        )
        candidates = fuse_three_way_ranks(
            candidates,
            colbert_scores,
            top_k=THREE_WAY_TOP_K,
        )
        rerank_scores = self.reranker.score(
            query, [candidate["retrieval_text"] for candidate in candidates]
        )
        ranked = fuse_scores_and_select_top_k(
            candidates,
            rerank_scores,
            coarse_weight=COARSE_WEIGHT,
            rerank_weight=RERANK_WEIGHT,
            top_k=len(candidates),
        )
        selected = filter_by_rerank_threshold(
            ranked, threshold=self.rerank_threshold, top_k=len(ranked)
        )

        expanded: list[dict[str, Any]] = []
        seen_parents: set[str] = set()
        for child in selected:
            if child["parent_id"] in seen_parents:
                continue
            expanded.append(self.searcher.expand_with_neighbors(child, window=1))
            seen_parents.add(child["parent_id"])
            if len(expanded) >= FINAL_TOP_K:
                break
        return expanded


def format_temporary_context(query: str, results: list[dict[str, Any]]) -> str:
    if not results:
        return query
    passages = []
    for rank, result in enumerate(results, 1):
        title = " > ".join(result.get("title_path", []))
        location = f"; Section: {title}" if title else ""
        pages = ""
        if result.get("page_start") is not None:
            pages = f"; Pages: {result['page_start']}-{result.get('page_end', result['page_start'])}"
        passages.append(
            f"[{rank}] Source: {result['source']}{location}{pages}\n"
            f"Combined relevance: {result['combined_score']:.4f}\n"
            f"{result['text']}"
        )
    return (
        f"{query}\n\n"
        "<temporary_retrieved_eeg_knowledge>\n"
        "Use the following retrieved EEG references only for this request. "
        "Each passage contains the matched child and its adjacent siblings. "
        "Treat them as supporting context, not as patient-specific findings.\n\n"
        + "\n\n".join(passages)
        + "\n</temporary_retrieved_eeg_knowledge>"
    )
