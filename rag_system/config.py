"""
RAG 系统配置 — 模型路径、默认参数、Prompt 模板。
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# 国内 HuggingFace 镜像（解决模型下载被墙问题）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# ── 路径 ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
JSONL_PATH = BASE_DIR / "斗罗大陆第一季.jsonl"
INDEX_DIR = BASE_DIR / "faiss_index"

# ── Embedding 模型 ────────────────────────────────────────────────────
# BGE-small-zh: 24MB，中文效果好，CPU 友好
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
EMBEDDING_DIM = 512  # bge-small-zh 输出 512 维

# ── 检索参数 ──────────────────────────────────────────────────────────
DEFAULT_TOP_K = 5        # 默认返回片段数
MAX_TOP_K = 20           # 最多返回
SIMILARITY_FLOOR = 0.3   # 相似度阈值，低于此值的结果丢弃

# ── LLM 配置（DeepSeek API，兼容 OpenAI SDK）─────────────────────────
LLM_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
LLM_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
LLM_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

# ── RAG Prompt 模板 ───────────────────────────────────────────────────
RAG_SYSTEM_PROMPT = """你是《斗罗大陆》小说知识问答助手。根据以下小说原文片段回答用户的问题。

规则：
1. 如果原文中包含答案，直接引用原文信息回答。
2. 如果原文中没有相关线索，明确说"小说中没有明确提到这个信息"。
3. 回答要简洁准确，不要编造情节。
4. 回答时尽量引用原文中的具体描述。"""

RAG_USER_PROMPT = """参考原文片段：
---
{context}
---

用户问题：{query}

请回答："""
