"""
《斗罗大陆》RAG 问答系统 — Streamlit UI
"""
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

import streamlit as st
from rag_system.pipeline import RAGPipeline


@st.cache_resource
def load_rag():
    """缓存 RAG 实例，避免重复加载模型。"""
    rag = RAGPipeline()
    rag.load_or_build()
    return rag


st.set_page_config(page_title="斗罗大陆 RAG 问答", page_icon="📖", layout="wide")
st.title("📖 斗罗大陆 RAG 知识问答")
st.caption("RAG (Retrieval-Augmented Generation) 检索增强生成 — 基于原文内容精准回答情节问题")

# 初始化
rag = load_rag()

# 搜索框
query = st.text_input(
    "输入你的问题",
    placeholder="例如：唐三的八蛛矛是什么时候获得的？",
    key="query_input",
)

col1, col2 = st.columns([1, 1])
with col1:
    top_k = st.slider("检索片段数", 3, 15, 5)
with col2:
    btn = st.button("搜索", type="primary", use_container_width=True)

# 问答逻辑
if btn and query.strip():
    with st.spinner("正在检索原文片段..."):
        result = rag.ask(query.strip(), top_k=top_k)

    st.markdown("---")
    st.markdown("### 回答")
    st.markdown(result["answer"])

    # 引用来源
    if result["sources"]:
        st.markdown("---")
        st.markdown("### 参考来源")
        for i, src in enumerate(result["sources"]):
            with st.expander(
                f"[{i+1}] 第{src['chapter_id']}章 {src['chapter_title']} "
                f"（相似度: {src['score']:.2f}）"
            ):
                st.text(src["snippet"])

elif btn and not query.strip():
    st.warning("请输入问题")
