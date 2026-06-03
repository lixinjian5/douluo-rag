"""
内容自动分类模块 —— 对每个 Chunk 标注其内容类型。

分类类型：
  - "对话"      :  引号内对话占比较高
  - "动作描写"  :  动作动词密度高
  - "环境描写"  :  环境描写词密度高 + 静态描述
  - "旁白"      :  默认 / 叙事 / 心理活动
"""
from .config import QUOTE_CHARS, QUOTE_PAIRS, KW_ACTION, KW_ENVIRONMENT


def _count_quote_content(text: str) -> int:
    """统计被成对中文引号包裹的字符数。"""
    total = 0
    for left, right in QUOTE_PAIRS:
        pos = 0
        while True:
            l = text.find(left, pos)
            if l == -1:
                break
            r = text.find(right, l + 1)
            if r == -1:
                break
            total += r - l - 1
            pos = r + 1
    return total


def _count_action_words(text: str) -> int:
    """统计动作动词出现次数。"""
    count = 0
    for kw in KW_ACTION:
        count += text.count(kw)
    return count


def _count_env_words(text: str) -> int:
    """统计环境描写词出现次数。"""
    count = 0
    for kw in KW_ENVIRONMENT:
        count += text.count(kw)
    return count


def classify_content(text: str) -> str:
    """
    对给定文本进行分类。

    策略：
    1. 先计算对话覆盖率：如果 >25% 的文本在引号内 → "对话"
    2. 再计算动作/环境密度：
       - 动作密度 > 环境密度 * 1.5 → "动作描写"
       - 环境密度 > 动作密度 * 1.5 → "环境描写"
    3. 其余 → "旁白"
    """
    total_len = len(text)
    if total_len == 0:
        return "旁白"

    # 对话占比
    quote_len = _count_quote_content(text)
    quote_ratio = quote_len / total_len

    if quote_ratio > 0.25:
        return "对话"

    # 动作 / 环境 密度
    action_count = _count_action_words(text)
    env_count = _count_env_words(text)

    if action_count == 0 and env_count == 0:
        return "旁白"

    if action_count > env_count * 1.5:
        return "动作描写"
    if env_count > action_count * 1.5:
        return "环境描写"

    return "旁白"
