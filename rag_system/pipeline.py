"""
RAG 主管线 — 串联 检索 + 生成，对外提供一个 ask() 入口。
"""
from .config import DEFAULT_TOP_K
from .embedder import Embedder
from .vector_store import VectorStore
from .retriever import Retriever
from .generator import Generator


class RAGPipeline:
    def __init__(self):
        self.embedder = Embedder()
        self.store = VectorStore()
        self.retriever = Retriever(self.store, self.embedder)
        self.generator = Generator()

    def load_or_build(self):
        """加载已有索引，若不存在则自动构建。"""
        if not self.store.load():
            print("索引不存在，开始构建...")
            self.store.build(embedder=self.embedder)

    def ask(self, query: str, top_k: int = DEFAULT_TOP_K) -> dict:
        """
        端到端问答入口。

        返回: {"answer": str, "sources": list[dict]}
        """
        results = self.retriever.retrieve(query, top_k=top_k)
        if not results:
            return {
                "answer": "未找到相关片段，无法回答此问题。",
                "sources": [],
            }

        context = self.retriever.format_context(results)
        answer = self.generator.generate(query, context)

        sources = [
            {
                "chapter_id": r.chapter_id,
                "chapter_title": r.chapter_title,
                "score": round(r.score, 4),
                "snippet": r.text[:150],
            }
            for r in results
        ]

        return {"answer": answer, "sources": sources}
