"""
Embedding 模型封装 — 加载本地模型，提供 embed() 和 embed_batch() 接口。
"""
import numpy as np
from sentence_transformers import SentenceTransformer
from .config import EMBEDDING_MODEL


class Embedder:
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model = SentenceTransformer(model_name)

    def embed(self, text: str) -> np.ndarray:
        """单条文本 embedding"""
        vec = self.model.encode(text, normalize_embeddings=True)
        return vec.astype(np.float32)

    def embed_batch(self, texts: list[str], show_progress: bool = True) -> np.ndarray:
        """批量 embedding"""
        vecs = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
            batch_size=64,
        )
        return vecs.astype(np.float32)
