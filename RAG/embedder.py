# 嵌入器模块。
from transformers import AutoTokenizer, AutoModel
import torch
from typing import List
import os

class BGEEmbedder:
    def __init__(self, model_name="sentenceModel/bge-m3"):
        """初始化对象状态。"""
        CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
        model_name = os.path.join(CURRENT_DIR, model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, texts: List[str], batch_size: int = 16) -> List[List[float]]:
        """处理 encode 相关逻辑。"""
        embeddings = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
            inputs = {name: value.to(self.device) for name, value in inputs.items()}
            outputs = self.model(**inputs)
            batch_embeddings = outputs.last_hidden_state[:, 0, :]
            batch_embeddings = torch.nn.functional.normalize(batch_embeddings, dim=1)
            embeddings.extend(batch_embeddings.cpu().tolist())
        return embeddings
