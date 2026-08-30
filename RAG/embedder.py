"""Local BGE-M3 dense, sparse, and ColBERT encoder."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from transformers import AutoModel, AutoTokenizer


class BGEEmbedder:
    def __init__(self, model_name: str = "sentenceModel/bge-m3", max_length: int = 1024):
        model_path = Path(__file__).resolve().parent / model_name
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
        self.model = AutoModel.from_pretrained(str(model_path), local_files_only=True)
        self.sparse_linear = torch.nn.Linear(self.model.config.hidden_size, 1)
        self.colbert_linear = torch.nn.Linear(
            self.model.config.hidden_size,
            self.model.config.hidden_size,
        )
        sparse_path = model_path / "sparse_linear.pt"
        colbert_path = model_path / "colbert_linear.pt"
        if not sparse_path.is_file():
            raise FileNotFoundError(f"BGE-M3 sparse weights not found: {sparse_path}")
        if not colbert_path.is_file():
            raise FileNotFoundError(f"BGE-M3 ColBERT weights not found: {colbert_path}")
        self.sparse_linear.load_state_dict(
            torch.load(sparse_path, map_location="cpu", weights_only=True)
        )
        self.colbert_linear.load_state_dict(
            torch.load(colbert_path, map_location="cpu", weights_only=True)
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.sparse_linear.to(self.device)
        self.colbert_linear.to(self.device)
        self.model.eval()
        self.sparse_linear.eval()
        self.colbert_linear.eval()
        self.max_length = max_length
        self.special_token_ids = set(self.tokenizer.all_special_ids)

    @torch.no_grad()
    def encode_hybrid(
        self,
        texts: Sequence[str],
        batch_size: int = 8,
        *,
        return_colbert: bool = False,
        max_length: int | None = None,
    ) -> dict[str, list]:
        dense_vectors: list[list[float]] = []
        lexical_weights: list[dict[int, float]] = []
        colbert_vectors: list[torch.Tensor] = []
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start:start + batch_size])
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length or self.max_length,
                return_tensors="pt",
            )
            inputs = {name: value.to(self.device) for name, value in inputs.items()}
            hidden = self.model(**inputs).last_hidden_state
            dense = torch.nn.functional.normalize(hidden[:, 0, :], dim=1)
            token_weights = torch.relu(self.sparse_linear(hidden).squeeze(-1))
            token_weights = token_weights * inputs["attention_mask"]
            if return_colbert:
                colbert = torch.nn.functional.normalize(
                    self.colbert_linear(hidden[:, 1:, :]), dim=-1
                )

            dense_vectors.extend(dense.cpu().tolist())
            for token_ids, weights, mask in zip(
                inputs["input_ids"].cpu().tolist(),
                token_weights.cpu().tolist(),
                inputs["attention_mask"].cpu().tolist(),
            ):
                sparse: dict[int, float] = {}
                for token_id, weight, keep in zip(token_ids, weights, mask):
                    if not keep or token_id in self.special_token_ids or weight <= 0:
                        continue
                    sparse[token_id] = max(sparse.get(token_id, 0.0), float(weight))
                lexical_weights.append(sparse)
            if return_colbert:
                for vectors, mask in zip(colbert, inputs["attention_mask"][:, 1:]):
                    colbert_vectors.append(vectors[mask.bool()].detach().cpu())
        result = {"dense_vecs": dense_vectors, "lexical_weights": lexical_weights}
        if return_colbert:
            result["colbert_vecs"] = colbert_vectors
        return result

    @torch.no_grad()
    def score_colbert(
        self,
        query_vector: torch.Tensor,
        passages: Sequence[str],
        batch_size: int = 4,
        max_passage_length: int = 512,
    ) -> list[float]:
        """Compute BGE-M3 late-interaction MaxSim scores for recalled children."""
        if not passages:
            return []
        query = query_vector.to(self.device)
        scores: list[float] = []
        for start in range(0, len(passages), batch_size):
            output = self.encode_hybrid(
                passages[start:start + batch_size],
                batch_size=batch_size,
                return_colbert=True,
                max_length=max_passage_length,
            )
            for passage_vector in output["colbert_vecs"]:
                passage = passage_vector.to(self.device)
                if query.numel() == 0 or passage.numel() == 0:
                    scores.append(0.0)
                    continue
                token_similarity = query @ passage.transpose(0, 1)
                scores.append(float(token_similarity.max(dim=1).values.mean().item()))
        return scores

    def encode(self, texts: Sequence[str], batch_size: int = 8) -> list[list[float]]:
        """Compatibility wrapper for callers that only need dense vectors."""
        return self.encode_hybrid(texts, batch_size=batch_size)["dense_vecs"]
