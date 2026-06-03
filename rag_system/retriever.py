"""
检索模块 — 将 query → embedding → FAISS 搜索 → 格式化上下文。
"""
from .config import DEFAULT_TOP_K
from .embedder import Embedder
from .vector_store import VectorStore, SearchResult


class Retriever:
    def __init__(self, vector_store: VectorStore, embedder: Embedder | None = None):
        self.store = vector_store
        self.embedder = embedder or Embedder()

    def retrieve(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[SearchResult]:
        """检索与 query 最相关的 Top-K 片段。"""
        query_vec = self.embedder.embed(query)
        return self.store.search(query_vec, top_k=top_k)

    def format_context(self, results: list[SearchResult]) -> str:
        """将检索结果拼接为 LLM 可直接使用的上下文字符串。"""
        parts = []
        for i, r in enumerate(results):
            header = f"[片段 {i+1}] 第{r.chapter_id}章 {r.chapter_title}"
            parts.append(f"{header}\n{r.text}")
        return "\n\n".join(parts)
