"""Local BGE reranker used for the second retrieval stage."""

from pathlib import Path
from typing import Sequence

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class BGEReranker:
    def __init__(self, model_path: str | Path | None = None, max_length: int = 1024):
        default_path = Path(__file__).resolve().parent / "sentenceModel" / "bge-reranker-v2-m3"
        self.model_path = Path(model_path or default_path).resolve()
        if not (self.model_path / "model.safetensors").is_file():
            raise FileNotFoundError(
                f"bge-reranker-v2-m3 weights not found: {self.model_path}"
            )
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path), local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(self.model_path),
            local_files_only=True,
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.max_length = max_length

    @torch.no_grad()
    def score(self, query: str, passages: Sequence[str], batch_size: int = 4) -> list[float]:
        """Return sigmoid-normalized relevance scores for query/passage pairs."""
        scores: list[float] = []
        for start in range(0, len(passages), batch_size):
            batch = passages[start:start + batch_size]
            pairs = [[query, passage] for passage in batch]
            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            inputs = {name: value.to(self.device) for name, value in inputs.items()}
            logits = self.model(**inputs).logits.reshape(-1).float()
            scores.extend(torch.sigmoid(logits).cpu().tolist())
        return scores
