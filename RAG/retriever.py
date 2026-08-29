"""Dynamic two-stage EEG knowledge retrieval with reranking."""

import os
from pathlib import Path
from typing import Any

from .embedder import BGEEmbedder
from .indexer import update_index
from .ranking import (
    filter_by_rerank_threshold,
    fuse_scores_and_select_top_k,
    passes_faiss_probe,
)
from .reranker import BGEReranker
from .searcher import FaissSearcher


FAISS_TOP_K = 20
FINAL_TOP_K = 3
COARSE_WEIGHT = 0.2
RERANK_WEIGHT = 0.8
DEFAULT_FAISS_PROBE_THRESHOLD = 0.35
DEFAULT_RERANK_THRESHOLD = 0.5


class EEGRetriever:
    def __init__(
        self,
        faiss_probe_threshold: float | None = None,
        rerank_threshold: float | None = None,
    ):
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
        self.searcher = FaissSearcher(
            str(self.rag_dir / "faiss.index"),
            str(self.rag_dir / "chunks.pkl"),
        )
        self.reranker = BGEReranker()

    def retrieve(
        self,
        query: str,
        *,
        require_faiss_probe: bool = False,
    ) -> list[dict[str, Any]]:
        query_vector = self.embedder.encode([query])[0]
        candidates = self.searcher.search_records(query_vector, top_k=FAISS_TOP_K)
        if not candidates:
            return []
        if require_faiss_probe and not passes_faiss_probe(
            candidates,
            self.faiss_probe_threshold,
        ):
            return []

        rerank_scores = self.reranker.score(
            query,
            [candidate["text"] for candidate in candidates],
        )
        ranked = fuse_scores_and_select_top_k(
            candidates,
            rerank_scores,
            coarse_weight=COARSE_WEIGHT,
            rerank_weight=RERANK_WEIGHT,
            top_k=len(candidates),
        )
        return filter_by_rerank_threshold(
            ranked,
            threshold=self.rerank_threshold,
            top_k=FINAL_TOP_K,
        )


def format_temporary_context(query: str, results: list[dict[str, Any]]) -> str:
    """Attach retrieved evidence to only the current user message."""
    if not results:
        return query
    passages = []
    for rank, result in enumerate(results, 1):
        passages.append(
            f"[{rank}] Source: {result['source']}\n"
            f"Combined relevance: {result['combined_score']:.4f}\n"
            f"{result['text']}"
        )
    return (
        f"{query}\n\n"
        "<temporary_retrieved_eeg_knowledge>\n"
        "Use the following retrieved EEG references only for this request. "
        "Treat them as supporting context, not as patient-specific findings.\n\n"
        + "\n\n".join(passages)
        + "\n</temporary_retrieved_eeg_knowledge>"
    )
