# 斗罗大陆 RAG 知识问答

基于 **RAG (Retrieval-Augmented Generation)** 架构的《斗罗大陆》小说问答系统。将 290 万字小说构建为向量知识库，通过**检索增强生成**实现基于原文的精准情节问答。

## RAG 架构

```
用户提问
   │
   ▼
┌──────────────────────────────────────┐
│  1. Embedding (BGE-small-zh, 512维)  │  ← Query 向量化
└──────────────────────────────────────┘
   │
   ▼
┌──────────────────────────────────────┐
│  2. Retrieval (FAISS 余弦检索 Top-K) │  ← 从 5,284 个 Chunk 中召回相关片段
└──────────────────────────────────────┘
   │
   ▼
┌──────────────────────────────────────┐
│  3. Generation (DeepSeek API)        │  ← 基于召回上下文生成答案
└──────────────────────────────────────┘
   │
   ▼
  答案 + 引用来源（章节、原文片段）
```

## 技术栈

| 环节 | 技术方案 |
|------|----------|
| **文本预处理** | Python 正则分章、语义分块、内容分类 |
| **向量化 (Embedding)** | BAAI/bge-small-zh-v1.5, 512 维, 本地 CPU 推理 |
| **向量检索 (Retrieval)** | FAISS IndexFlatIP, 余弦相似度, Top-K 召回 |
| **答案生成 (Generation)** | DeepSeek API, temperature=0.3 |

## 项目结构

```
douluo-rag/
├── douluo_rag/              # 预处理管线
│   ├── config.py            # 清洗参数、中文数字映射
│   ├── cleaner.py           # 文本清洗、广告过滤
│   ├── splitter.py          # 章节识别（正则 + 中文数字解析）
│   ├── chunker.py           # 语义分块（事件锚点 + 重叠窗口）
│   ├── classifier.py        # 内容分类（对话/动作/环境/旁白）
│   └── pipeline.py          # 预处理主流程
│
├── rag_system/              # RAG 核心模块
│   ├── embedder.py          # BGE Embedding（本地 CPU）
│   ├── vector_store.py      # FAISS 索引构建 / 搜索
│   ├── retriever.py         # Query → Embedding → 检索 → 格式化
│   ├── generator.py         # DeepSeek 答案生成
│   ├── config.py            # 模型路径、检索参数、Prompt
│   └── pipeline.py          # RAGPipeline.ask() 端到端入口
│
├── app.py                   # Streamlit 问答 UI
├── build_index.py           # FAISS 索引构建
├── faiss_index/             # 向量索引文件
├── 斗罗大陆第一季.jsonl      # 预处理输出 (5,284 Chunks)
└── .env.template            # API Key 配置模板
```

## RAG 关键参数

| 参数 | 值 |
|------|-----|
| 分块大小 | 500-750 字符，100 字符重叠 |
| Embedding 维度 | 512 (BGE-small-zh) |
| 检索 Top-K | 5 (可调) |
| 相似度阈值 | 0.3 |
| LLM Temperature | 0.3 |

## 快速开始

```bash
# 1. 安装依赖
pip install streamlit openai python-dotenv sentence-transformers faiss-cpu

# 2. 配置 API Key
cp .env.template .env
# 编辑 .env，填入你的 DeepSeek API Key

# 3. 构建索引（首次运行）
python build_index.py

# 4. 启动问答 UI
streamlit run app.py
```

## 预处理数据

- 原始小说：310 章，290 万字符
- 预处理后：5,284 个语义 Chunk
- 内容分类：动作 60% / 对话 38% / 旁白+环境 2%

## 注意事项

- 原始文件编码为 **GBK**，非 UTF-8
- HuggingFace 国内用户需设置镜像：`HF_ENDPOINT=https://hf-mirror.com`
- `.env` 包含 API Key，已加入 `.gitignore`，请勿提交
