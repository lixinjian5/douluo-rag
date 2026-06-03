"""一次性索引构建脚本 — 使用 HF 镜像下载模型。"""
import os

# 国内用户：使用 HuggingFace 镜像
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from rag_system.vector_store import VectorStore

store = VectorStore()
store.build()
