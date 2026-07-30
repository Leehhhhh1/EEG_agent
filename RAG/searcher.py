# 检索器模块。
import faiss
import pickle
import numpy as np

class FaissSearcher:
    def __init__(self, index_path: str, text_path: str):
        """初始化对象状态。"""
        self.index = faiss.read_index(index_path)
        with open(text_path, 'rb') as f:
            self.texts = pickle.load(f)

    def search(self, query_vector, top_k=5):
        """处理 search 相关逻辑。"""
        D, I = self.index.search(np.array([query_vector]).astype('float32'), top_k)
        results = [(self.texts[i], float(D[0][j])) for j, i in enumerate(I[0])]
        return results
