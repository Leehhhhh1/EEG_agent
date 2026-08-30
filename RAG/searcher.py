"""Dense and sparse retrieval over child chunks."""

from __future__ import annotations

from collections import defaultdict
import pickle

import faiss
import numpy as np

from .chunker import merge_child_texts


class HybridSearcher:
    def __init__(self, index_path: str, children_path: str, parents_path: str, sparse_path: str):
        self.index = faiss.read_index(index_path)
        with open(children_path, "rb") as file:
            self.children: list[dict] = pickle.load(file)
        with open(parents_path, "rb") as file:
            self.parents: dict[str, dict] = pickle.load(file)
        with open(sparse_path, "rb") as file:
            self.sparse_index: dict[int, list[tuple[int, float]]] = pickle.load(file)
        self.children_by_parent: dict[str, list[dict]] = defaultdict(list)
        for child in self.children:
            self.children_by_parent[child["parent_id"]].append(child)

    def search_dense(self, query_vector, top_k: int = 30) -> list[dict]:
        available = min(top_k, self.index.ntotal)
        if available <= 0:
            return []
        distances, indexes = self.index.search(
            np.asarray([query_vector], dtype="float32"), available
        )
        results = []
        for position, child_index in enumerate(indexes[0]):
            if child_index < 0:
                continue
            item = dict(self.children[int(child_index)])
            item["dense_score"] = float(distances[0][position])
            results.append(item)
        return results

    def search_sparse(self, query_weights: dict[int, float], top_k: int = 30) -> list[dict]:
        scores: dict[int, float] = defaultdict(float)
        for token_id, query_weight in query_weights.items():
            for child_index, document_weight in self.sparse_index.get(int(token_id), []):
                scores[int(child_index)] += float(query_weight) * float(document_weight)
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
        results = []
        for child_index, score in ranked:
            item = dict(self.children[child_index])
            item["sparse_score"] = score
            results.append(item)
        return results

    def expand_with_neighbors(self, child: dict, window: int = 1) -> dict:
        siblings = sorted(
            self.children_by_parent[child["parent_id"]],
            key=lambda item: int(item["child_index"]),
        )
        position = next(
            index for index, item in enumerate(siblings)
            if item["chunk_id"] == child["chunk_id"]
        )
        selected = siblings[max(0, position - window):position + window + 1]
        parent = self.parents.get(child["parent_id"], {})
        result = dict(child)
        result["text"] = merge_child_texts(selected)
        result["matched_text"] = child["text"]
        result["context_chunk_ids"] = [item["chunk_id"] for item in selected]
        result["page_start"] = parent.get("page_start", child.get("page_start"))
        result["page_end"] = parent.get("page_end", child.get("page_end"))
        return result


FaissSearcher = HybridSearcher
