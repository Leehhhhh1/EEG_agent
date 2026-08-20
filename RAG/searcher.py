# 检索器模块。
import faiss
import pickle
import numpy as np

class FaissSearcher:
    def __init__(self, index_path: str, text_path: str):
        """初始化对象状态。"""
        self.index = faiss.read_index(index_path)
        with open(text_path, 'rb') as f:
            self.records = pickle.load(f)

    def search_records(self, query_vector, top_k=20):
        """Return metadata-bearing FAISS candidates with their coarse scores."""
        available = min(top_k, self.index.ntotal)
        if available <= 0:
            return []
        distances, indexes = self.index.search(
            np.array([query_vector]).astype('float32'),
            available,
        )
        results = []
        for position, record_index in enumerate(indexes[0]):
            if record_index < 0:
                continue
            record = self.records[record_index]
            if not isinstance(record, dict):
                raise ValueError("The FAISS text store uses the legacy format; rebuild the RAG index.")
            result = dict(record)
            result["coarse_score"] = float(distances[0][position])
            results.append(result)
        return results

    def search(self, query_vector, top_k=5):
        """Compatibility wrapper returning the historical tuple format."""
        return [
            (record["text"], record["coarse_score"])
            for record in self.search_records(query_vector, top_k=top_k)
        ]
