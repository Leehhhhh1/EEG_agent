# 索引器模块。
import faiss
import pickle
import numpy as np
import os
import json
from .chunker import load_and_chunk
from .embedder import BGEEmbedder

class FaissIndexer:
    def __init__(self, dim: int):
        """初始化对象状态。"""
        self.index = faiss.IndexFlatIP(dim)
        self.records = []

    def add(self, vectors, records):
        """处理 add 相关逻辑。"""
        self.index.add(np.array(vectors).astype('float32'))
        self.records.extend(records)

    def save(self, index_path: str, text_path: str):
        """处理 save 相关逻辑。"""
        faiss.write_index(self.index, index_path)
        with open(text_path, 'wb') as f:
            pickle.dump(self.records, f)

    def load(self, index_path: str, text_path: str):
        """加载 load 所需的数据。"""
        self.index = faiss.read_index(index_path)
        with open(text_path, 'rb') as f:
            self.records = pickle.load(f)

    def reset(self):
        """重置当前对象的内部状态。"""
        self.index = faiss.IndexFlatIP(self.index.d)
        self.records = []

def _document_registry(docs_dir: str) -> dict[str, float]:
    """Return a stable registry keyed by source filename."""
    return {
        filename: os.path.getmtime(os.path.join(docs_dir, filename))
        for filename in sorted(os.listdir(docs_dir))
        if os.path.isfile(os.path.join(docs_dir, filename))
        and filename.lower().endswith((".pdf", ".txt"))
    }


def _has_metadata_records(text_path: str) -> bool:
    if not os.path.exists(text_path):
        return False
    try:
        with open(text_path, "rb") as file:
            records = pickle.load(file)
        return bool(records) and all(
            isinstance(record, dict) and {"text", "source"} <= record.keys()
            for record in records
        )
    except (OSError, pickle.PickleError, EOFError):
        return False


def update_index(
    docs_dir="docs",
    index_path="faiss.index",
    text_path="chunks.pkl",
    registry_path="registered_files.json",
    embedder=None,
):
    """处理 update index 相关逻辑。"""
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    docs_dir = os.path.join(CURRENT_DIR, docs_dir)
    index_path = os.path.join(CURRENT_DIR, index_path)
    text_path = os.path.join(CURRENT_DIR, text_path)
    registry_path = os.path.join(CURRENT_DIR, registry_path)

    current_registry = _document_registry(docs_dir)
    if os.path.exists(registry_path):
        with open(registry_path, "r", encoding="utf-8") as f:
            registered_files = json.load(f)
    else:
        registered_files = {}

    if (
        registered_files == current_registry
        and os.path.exists(index_path)
        and _has_metadata_records(text_path)
    ):
        return False

    embedder = embedder or BGEEmbedder()
    indexer = FaissIndexer(dim=1024)
    for filename in sorted(current_registry):
        filepath = os.path.join(docs_dir, filename)
        print(f"Processing new file: {filepath}")
        chunks = load_and_chunk(filepath)
        if not chunks:
            continue
        vectors = embedder.encode(chunks)
        records = [
            {
                "text": text,
                "source": filename,
                "chunk_id": f"{filename}:{chunk_index}",
            }
            for chunk_index, text in enumerate(chunks)
        ]
        indexer.add(vectors, records)

    if not indexer.records:
        raise ValueError(f"No PDF or TXT chunks were found in {docs_dir}.")
    indexer.save(index_path, text_path)
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(current_registry, f, ensure_ascii=False, indent=2)

    print("Index update complete.")
    return True
