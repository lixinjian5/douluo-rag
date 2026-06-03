# 斗罗大陆 RAG 问答系统

## 项目概述
将《斗罗大陆》小说 (2.9M 字符, GBK) 处理为 RAG 知识库，支持基于原文的精准情节问答。

技术栈: Python + BGE-small-zh (本地 Embedding) + FAISS (向量检索) + DeepSeek API (生成)

## 项目结构

```
douluo/
├── douluo_rag/              # 预处理模块
│   ├── config.py            # 清洗参数、中文数字映射、分类词库
│   ├── cleaner.py           # 文本清洗、广告过滤、后记截断
│   ├── splitter.py          # 章节识别 (正则 + 中文数字解析)
│   ├── chunker.py           # 语义分块 (句子边界 + 事件锚点 + 重叠窗口)
│   ├── classifier.py        # 内容分类 (对话/动作/环境/旁白)
│   └── pipeline.py          # 预处理主流程
│
├── rag_system/              # RAG 问答模块
│   ├── config.py            # 模型路径、检索参数、Prompt 模板
│   ├── embedder.py          # BGE-small-zh-v1.5 (512维, 本地 CPU)
│   ├── vector_store.py      # FAISS IndexFlatIP 构建/搜索
│   ├── retriever.py         # query → embedding → search → format context
│   ├── generator.py         # DeepSeek API 答案生成
│   └── pipeline.py          # RAGPipeline.ask() 端到端入口
│
├── 斗罗大陆第一季.txt        # 原始小说 (GBK 编码)
├── 斗罗大陆第一季.jsonl      # 预处理输出 5,284 条 Chunk
├── faiss_index/             # FAISS 索引 (10.3MB)
├── build_index.py           # 一次性索引构建
├── test_pipeline.py         # 预处理模块测试
├── app.py                   # Streamlit UI
├── .env.template            # API Key 配置模板
└── .env                     # 实际 API Key (不提交)
```

## 关键参数

- **分块**: 500-750 字符/块, 100 字符重叠, 事件锚点优先切分
- **Embedding**: BAAI/bge-small-zh-v1.5, 512 维, HF 镜像 hf-mirror.com
- **检索**: Top-K=5, 相似度阈值 0.3
- **LLM**: DeepSeek-chat, temperature=0.3
- **预处理输出**: 310 章, 5,284 个 Chunk, 分类: 动作 60% / 对话 38% / 旁白+环境 2%

## 常用命令

```bash
# 预处理 (已执行，一般不需重跑)
python douluo_rag/pipeline.py

# 构建 FAISS 索引 (已执行)
python build_index.py

# 测试预处理管线
python test_pipeline.py

# 启动问答 UI
streamlit run app.py
```

## RAG 问答流程

```
用户提问 → BGE embedding (512维) → FAISS 余弦搜索 Top-K
         → 拼接上下文 → DeepSeek 基于原文生成答案 + 引用来源
```

## 注意事项

- 原始文件编码为 GBK，不是 UTF-8
- HuggingFace 国内需走镜像 (HF_ENDPOINT=https://hf-mirror.com)，已在 config.py 中设置
- .env 包含 DeepSeek API Key，不要提交到 Git
- 环境: Python 3.14, Windows 11
