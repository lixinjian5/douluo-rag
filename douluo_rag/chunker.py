"""
语义级智能分块模块 —— 按句子边界切分，在事件锚点处优先分割，
控制块大小在 CHUNK_MIN_CHARS~CHUNK_MAX_CHARS 之间，相邻块有重叠。
"""
import re
from .config import (
    SENTENCE_END, KW_TIME_SKIP, KW_SCENE_CHANGE,
    CHUNK_MIN_CHARS, CHUNK_MAX_CHARS, OVERLAP_CHARS, estimate_tokens,
)


def is_event_anchor(sentence: str) -> bool:
    """
    判断句子是否以"事件锚点"开头 —— 即时间跳跃词或场景转换词。
    这些位置是语义边界，应优先在此处分块。
    """
    head = sentence[:6].strip()
    for kw in KW_TIME_SKIP:
        if head.startswith(kw):
            return True
    for kw in KW_SCENE_CHANGE:
        if head.startswith(kw):
            return True
    return False


def split_sentences(text: str) -> list[str]:
    """
    按中文句子结束标点切分文本，保留标点附着在句尾。
    SENTENCE_END 匹配 。！？…— 」』" ）等标点后紧跟非空白字符的位置。
    """
    # 使用 re.split 但保留分隔符。更稳健的做法：在匹配处插入特殊标记再切分
    # 这里使用简单方法：在匹配位置后插入换行，再按换行切分
    # 更稳妥的方式：手动遍历匹配位置
    parts = []
    last_end = 0
    for m in SENTENCE_END.finditer(text):
        pos = m.end()
        sentence = text[last_end:pos].strip()
        if sentence and len(sentence) >= 2:  # 过滤空句和单字符
            parts.append(sentence)
        last_end = pos
    # 尾部剩余
    tail = text[last_end:].strip()
    if tail and len(tail) >= 2:
        parts.append(tail)

    return parts


def _score_split_point(sentence: str, current_size: int) -> float:
    """
    给当前拆分点打分，分数越高越适合作为分块边界。

    评分规则：
    - 事件锚点开头：+10 分
    - 已超 max：事件锚点 +5，普通句 -5
    - 使用分割线 "-----" 之类的：+8
    """
    score = 0.0
    if is_event_anchor(sentence):
        score += 10
    if current_size > CHUNK_MAX_CHARS:
        score += 3  # 强制倾向切开
    if current_size < CHUNK_MIN_CHARS:
        score -= 3  # 还没到最小尺寸，不急着切
    # 分割线标记
    if re.match(r"^-{3,}\s*$", sentence):
        score += 8
    # 场景转换词在中间也加分
    head = sentence[:6]
    if any(head.startswith(kw) for kw in KW_SCENE_CHANGE):
        score += 5

    return score


def create_chunks(
    text: str,
    min_chars: int = CHUNK_MIN_CHARS,
    max_chars: int = CHUNK_MAX_CHARS,
    overlap_chars: int = OVERLAP_CHARS,
) -> list[str]:
    """
    核心分块函数。

    算法：
    1. 将文本按句子边界切分为句子列表
    2. 贪心合并句子，当累积长度接近或超过 max_chars 时寻找最佳切分点
    3. 在事件锚点处优先切分
    4. 相邻块之间保留 overlap_chars 长度的重叠文本

    Args:
        text: 待分块的文本
        min_chars: 最小块大小（字符数）
        max_chars: 最大块大小（字符数）
        overlap_chars: 重叠字符数

    Returns:
        分块后的文本列表
    """
    sentences = split_sentences(text)
    if not sentences:
        return []

    chunks = []
    i = 0

    while i < len(sentences):
        chunk_parts = []
        current_len = 0
        j = i

        while j < len(sentences):
            sent = sentences[j]
            sent_len = len(sent)

            # 如果加上这句还在 max 范围内，直接加入
            if current_len + sent_len <= max_chars:
                chunk_parts.append(sent)
                current_len += sent_len
                j += 1
                # 如果达到 min 且下句是事件锚点 → 在此切分
                if current_len >= min_chars and j < len(sentences) and is_event_anchor(sentences[j]):
                    break
            else:
                # 超过了 max_chars
                if current_len >= min_chars:
                    # 已经够大了，在此切分
                    # 但优先在事件锚点处切
                    if is_event_anchor(sent):
                        # 即使加这句会超 max，也在之前切掉
                        pass  # j 不增加，在此处结束
                    break
                else:
                    # 还不够 min，但单句超过 max → 强制加入（超长句只能保留）
                    chunk_parts.append(sent)
                    current_len += sent_len
                    j += 1
                break

        if chunk_parts:
            chunk_text = "".join(chunk_parts)
            chunks.append(chunk_text)

        # 设置下一个块的起点：重叠策略
        # 从当前块的结尾往回找 overlap_chars，找到该位置所在的句子作为下一块的起点
        if j < len(sentences):
            # 向后回溯 overlap，至少从 j-1 开始
            overlap_target = max(0, current_len - overlap_chars)
            backtrack_len = 0
            next_i = j
            for k in range(len(chunk_parts) - 1, -1, -1):
                backtrack_len += len(chunk_parts[k])
                next_i = i + k
                if backtrack_len >= overlap_chars:
                    break
            i = max(i + 1, next_i)  # 至少前进 1 句，避免死循环
        else:
            i = j  # 结束

    return chunks
