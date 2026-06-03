"""
章节切分模块 —— 识别章节标题，提取章节元数据，将文本按章节边界拆分。
"""
import re
from dataclasses import dataclass, field
from .config import RE_CHAPTER, CN_NUM_MAP, CN_NUM_UNITS


@dataclass
class Chapter:
    chapter_id: str       # "297"
    chapter_title: str     # "小舞献祭"
    section: str           # "" / "一" / "二" ...
    content: str           # 该章节正文（清洗后）


def _cn_num_to_int(cn: str) -> int:
    """将中文数字字符串转为整数。支持 '三百三十六' → 336, '二十一' → 21。"""
    if not cn:
        return 0
    if cn.isdigit():
        return int(cn)

    result = 0
    current = 0
    for ch in cn:
        if ch in CN_NUM_UNITS:
            unit = CN_NUM_UNITS[ch]
            if current == 0:
                current = 1
            result += current * unit
            current = 0
        else:
            val = CN_NUM_MAP.get(ch)
            if val is not None:
                current = current * 10 + val if current > 0 else val

    result += current
    return result


def extract_chapter_metadata(title_line: str) -> tuple[str, str, str]:
    """
    从章节标题行提取 (chapter_id, chapter_title, section)。

    示例：
      "第一章 斗罗大陆异界唐三(一)" → ("1", "斗罗大陆异界唐三", "一")
      "第三十六章 魂尊(三)"       → ("36", "魂尊", "三")
      "第297章 小舞献祭"          → ("297", "小舞献祭", "")
    """
    m = RE_CHAPTER.match(title_line.strip())
    if not m:
        return ("0", title_line, "")

    cn_num = m.group(1)
    raw_title = m.group(2).strip()
    section = m.group(3) or ""

    # 清洗标题尾部多余空格/标点
    raw_title = re.sub(r"\s+", "", raw_title)
    raw_title = raw_title.rstrip("（(")

    chapter_id = str(_cn_num_to_int(cn_num))
    return (chapter_id, raw_title, section)


def find_chapter_positions(text: str) -> list[tuple[int, str, str, str]]:
    """
    找出所有章节标题位置。

    返回 [(起始位置, chapter_id, chapter_title, section), ...]

    去重策略：同一章的多节标题连续出现时保留全部（内容不同），
    但如果完全相同的标题连续出现则跳过后续重复。
    """
    positions = []
    prev_id_section = None

    for m in RE_CHAPTER.finditer(text):
        start = m.start()
        full_match = m.group(0)
        chapter_id, chapter_title, section = extract_chapter_metadata(full_match)

        # 跳过紧邻的完全相同的章节（重复标题）
        current_key = (chapter_id, section)
        if current_key == prev_id_section and positions:
            # 这个位置紧邻上一处 → 可能是重复标题
            if start - positions[-1][0] < len(full_match) + 20:
                continue
        prev_id_section = current_key
        positions.append((start, chapter_id, chapter_title, section))

    return positions


def split_by_chapters(text: str) -> list[Chapter]:
    """
    将文本按章节边界切分为 Chapter 对象列表。

    第一个章节之前的内容（序言）也作为一个特殊 Chapter（chapter_id="_prologue"）。
    正文末尾的非章节内容会被丢弃。
    """
    positions = find_chapter_positions(text)
    if not positions:
        return [Chapter(chapter_id="_full", chapter_title="全文", section="", content=text)]

    chapters = []

    # 序言：第一个章节标记之前的内容
    first_pos = positions[0][0]
    prologue = text[:first_pos].strip()
    if prologue and len(prologue) > 50:
        chapters.append(Chapter(
            chapter_id="_prologue",
            chapter_title="序章",
            section="",
            content=prologue,
        ))

    # 各章节
    for i, (start, ch_id, ch_title, section) in enumerate(positions):
        content_start = start + len(RE_CHAPTER.match(text[start:]).group(0)) if RE_CHAPTER.match(text[start:]) else start
        content_start = text.index("\n", content_start) + 1 if "\n" in text[content_start:content_start + 10] else content_start

        if i + 1 < len(positions):
            content_end = positions[i + 1][0]
        else:
            content_end = len(text)

        content = text[content_start:content_end].strip()
        if content:
            chapters.append(Chapter(
                chapter_id=ch_id,
                chapter_title=ch_title,
                section=section,
                content=content,
            ))

    return chapters
