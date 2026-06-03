"""
向量库模块 — 基于 FAISS 的索引构建、保存、加载和搜索。
"""
import json
import numpy as np
import faiss
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

from .config import INDEX_DIR, JSONL_PATH, DEFAULT_TOP_K, SIMILARITY_FLOOR
from .embedder import Embedder


@dataclass
class SearchResult:
    chunk_id: str
    chapter_id: str
    chapter_title: str
    content_type: str
    text: str
    score: float  # cosine similarity


class VectorStore:
    def __init__(self, index_dir: Path = INDEX_DIR):
        self.index_dir = Path(index_dir)
        self.index: Optional[faiss.IndexFlatIP] = None
        self.records: list[dict] = []
        self._loaded = False

    # ── 构建索引 ──────────────────────────────────────────────────────

    def build(self, jsonl_path: Path = JSONL_PATH, embedder: Embedder | None = None):
        """从 JSONL 文件构建 FAISS 索引并持久化。"""
        if embedder is None:
            embedder = Embedder()

        print(f"加载 JSONL: {jsonl_path}")
        with open(jsonl_path, "r", encoding="utf-8") as f:
            self.records = [json.loads(line) for line in f if line.strip()]
        print(f"共 {len(self.records)} 条记录")

        print("生成 embedding...")
        texts = [r["text"] for r in self.records]
        vectors = embedder.embed_batch(texts)

        print(f"构建 FAISS 索引 ({vectors.shape[1]} 维)...")
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)

        self._save()
        print(f"索引已保存到 {self.index_dir}")

    # ── 加载索引 ──────────────────────────────────────────────────────

    def load(self) -> bool:
        """加载已持久化的 FAISS 索引。返回 False 表示索引不存在。"""
        index_path = self.index_dir / "index.faiss"
        records_path = self.index_dir / "records.json"

        if not index_path.exists() or not records_path.exists():
            self._loaded = False
            return False

        self.index = faiss.read_index(str(index_path))
        with open(records_path, "r", encoding="utf-8") as f:
            self.records = json.load(f)

        self._loaded = True
        print(f"索引已加载: {self.index.ntotal} 条向量, {len(self.records)} 条记录")
        return True

    def _save(self):
        """持久化索引和元数据到磁盘。"""
        self.index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_dir / "index.faiss"))
        with open(self.index_dir / "records.json", "w", encoding="utf-8") as f:
            json.dump(self.records, f, ensure_ascii=False, indent=2)

    # ── 搜索 ──────────────────────────────────────────────────────────

    def search(
        self,
        query_vec: np.ndarray,
        top_k: int = DEFAULT_TOP_K,
        floor: float = SIMILARITY_FLOOR,
    ) -> list[SearchResult]:
        if self.index is None:
            raise RuntimeError("索引未加载，请先 build() 或 load()")

        query_vec = query_vec.reshape(1, -1).astype(np.float32)
        scores, indices = self.index.search(query_vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or score < floor:
                continue
            rec = self.records[idx]
            results.append(SearchResult(
                chunk_id=rec["chunk_id"],
                chapter_id=rec["chapter_id"],
                chapter_title=rec["chapter_title"],
                content_type=rec["content_type"],
                text=rec["text"],
                score=float(score),
            ))
        return results

    @property
    def is_loaded(self) -> bool:
        return self._loaded and self.index is not None
