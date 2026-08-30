"""Build versioned parent/child dense and sparse RAG indexes."""

from __future__ import annotations

import json
from pathlib import Path
import pickle

import faiss
import numpy as np

from .chunker import load_and_chunk
from .docling_parser import SUPPORTED_DOCUMENT_EXTENSIONS
from .embedder import BGEEmbedder


INDEX_VERSION = 3
PARSER_CONFIG = {
    "engine": "docling",
    "heading_hierarchy": True,
    "ocr": True,
    "table_structure": True,
}
CHUNKING_CONFIG = {
    "parent_target_tokens": 900,
    "parent_max_tokens": 1400,
    "child_target_tokens": 280,
    "child_max_tokens": 400,
    "child_overlap_tokens": 50,
}


class FaissIndexer:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatIP(dim)
        self.records: list[dict] = []

    def add(self, vectors, records):
        self.index.add(np.asarray(vectors, dtype="float32"))
        self.records.extend(records)

    def save(self, index_path: str):
        faiss.write_index(self.index, index_path)


def _document_registry(docs_dir: Path) -> dict:
    files = {}
    for path in sorted(docs_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_DOCUMENT_EXTENSIONS:
            stat = path.stat()
            files[path.name] = {"mtime_ns": stat.st_mtime_ns, "size": stat.st_size}
    return {
        "index_version": INDEX_VERSION,
        "model": "bge-m3-dense+sparse",
        "parser": PARSER_CONFIG,
        "chunking": CHUNKING_CONFIG,
        "files": files,
    }


def _load_registry(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _save_pickle(path: Path, value) -> None:
    with path.open("wb") as file:
        pickle.dump(value, file, protocol=pickle.HIGHEST_PROTOCOL)


def _build_sparse_index(sparse_vectors: list[dict[int, float]]) -> dict[int, list[tuple[int, float]]]:
    inverted: dict[int, list[tuple[int, float]]] = {}
    for child_index, weights in enumerate(sparse_vectors):
        for token_id, weight in weights.items():
            inverted.setdefault(int(token_id), []).append((child_index, float(weight)))
    return inverted


def update_index(
    docs_dir: str = "docs",
    index_path: str = "faiss.index",
    children_path: str = "children.pkl",
    parents_path: str = "parents.pkl",
    sparse_path: str = "sparse_index.pkl",
    registry_path: str = "registered_files.json",
    embedder=None,
) -> bool:
    rag_dir = Path(__file__).resolve().parent
    docs = rag_dir / docs_dir
    paths = {
        "index": rag_dir / index_path,
        "children": rag_dir / children_path,
        "parents": rag_dir / parents_path,
        "sparse": rag_dir / sparse_path,
        "registry": rag_dir / registry_path,
    }
    current_registry = _document_registry(docs)
    required = [paths["index"], paths["children"], paths["parents"], paths["sparse"]]
    if _load_registry(paths["registry"]) == current_registry and all(path.exists() for path in required):
        return False

    embedder = embedder or BGEEmbedder()
    parents: list[dict] = []
    children: list[dict] = []
    for filename in sorted(current_registry["files"]):
        filepath = docs / filename
        print(f"Processing knowledge file: {filepath}")
        document_parents, document_children = load_and_chunk(
            str(filepath), tokenizer=embedder.tokenizer
        )
        parents.extend(document_parents)
        children.extend(document_children)

    if not children:
        raise ValueError(f"No Docling-supported knowledge documents were found in {docs}.")
    retrieval_texts = [child["retrieval_text"] for child in children]
    encoded = embedder.encode_hybrid(retrieval_texts)
    dense_vectors = encoded["dense_vecs"]
    sparse_vectors = encoded["lexical_weights"]

    indexer = FaissIndexer(dim=len(dense_vectors[0]))
    indexer.add(dense_vectors, children)
    indexer.save(str(paths["index"]))
    _save_pickle(paths["children"], children)
    _save_pickle(paths["parents"], {parent["parent_id"]: parent for parent in parents})
    _save_pickle(paths["sparse"], _build_sparse_index(sparse_vectors))
    paths["registry"].write_text(
        json.dumps(current_registry, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Index update complete: {len(parents)} parents, {len(children)} children.")
    return True
