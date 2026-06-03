"""
生成模块 — 调用 LLM 基于检索到的上下文生成答案。
"""
import os
from openai import OpenAI
from .config import LLM_BASE_URL, LLM_API_KEY, LLM_MODEL, RAG_SYSTEM_PROMPT, RAG_USER_PROMPT


class Generator:
    def __init__(
        self,
        base_url: str = LLM_BASE_URL,
        api_key: str = LLM_API_KEY,
        model: str = LLM_MODEL,
    ):
        self.model = model
        key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not key:
            raise ValueError("DEEPSEEK_API_KEY 未设置，请检查 .env 文件")
        self.client = OpenAI(base_url=base_url, api_key=key)

    def generate(self, query: str, context: str) -> str:
        """基于上下文生成回答。"""
        user_prompt = RAG_USER_PROMPT.format(context=context, query=query)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": RAG_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=800,
        )
        return response.choices[0].message.content
