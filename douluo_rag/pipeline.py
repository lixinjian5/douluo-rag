"""
主流程编排 —— 将清洗、章节切分、语义分块、分类、输出 JSONL 串联起来。
"""
import json
from pathlib import Path
from typing import Iterator

from .cleaner import clean_text, remove_non_content, trim_author_postscript
from .splitter import split_by_chapters, Chapter
from .chunker import create_chunks
from .classifier import classify_content
from .config import estimate_tokens


def _build_chunk_record(
    chunk_text: str,
    chapter_id: str,
    chapter_title: str,
    chunk_idx: int,
) -> dict:
    """为单个 Chunk 构建完整的 JSON 记录。"""
    return {
        "chunk_id": f"ch{chapter_id}_{chunk_idx:04d}",
        "chapter_id": chapter_id,
        "chapter_title": chapter_title,
        "char_count": len(chunk_text),
        "token_estimate": estimate_tokens(chunk_text),
        "content_type": classify_content(chunk_text),
        "text": chunk_text,
    }


def process_chapter(chapter: Chapter) -> Iterator[dict]:
    """对单个章节执行分块、分类，返回 JSON 记录迭代器。"""
    chunks = create_chunks(chapter.content)
    for i, chunk in enumerate(chunks):
        yield _build_chunk_record(chunk, chapter.chapter_id, chapter.chapter_title, i)


def process_text(raw_text: str) -> list[dict]:
    """处理内存中的文本，返回所有 Chunk 记录列表。"""
    text = clean_text(raw_text)
    text = remove_non_content(text)
    text = trim_author_postscript(text)
    chapters = split_by_chapters(text)

    all_records = []
    for ch in chapters:
        all_records.extend(process_chapter(ch))

    return all_records


def process_file(input_path: str | Path, output_path: str | Path | None = None) -> list[dict]:
    """
    从文件读取 → 处理 → 写入 JSONL。

    Args:
        input_path: 输入 TXT 文件路径 (GBK 编码)
        output_path: 输出 JSONL 路径，为 None 则自动生成

    Returns:
        所有 Chunk 记录列表
    """
    input_path = Path(input_path)

    # 尝试多种编码
    content = None
    for encoding in ["gbk", "gb18030", "gb2312", "utf-8"]:
        try:
            with open(input_path, "r", encoding=encoding) as f:
                content = f.read()
            break
        except (UnicodeDecodeError, UnicodeError):
            continue

    if content is None:
        raise ValueError(f"无法识别文件编码: {input_path}")

    records = process_text(content)

    # 写入 JSONL
    if output_path is None:
        output_path = input_path.with_suffix(".jsonl")
    else:
        output_path = Path(output_path)

    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # 统计信息
    print(f"输入: {input_path} ({len(content):,} 字符)")
    print(f"输出: {output_path} ({len(records):,} 个 Chunk)")
    print(f"章节数: {len(set(r['chapter_id'] for r in records))}")

    # 分类分布
    type_dist = {}
    for r in records:
        t = r["content_type"]
        type_dist[t] = type_dist.get(t, 0) + 1
    print(f"分类分布: {type_dist}")

    # 分块尺寸统计
    sizes = [r["char_count"] for r in records]
    print(f"块尺寸: min={min(sizes)}, max={max(sizes)}, avg={sum(sizes)/len(sizes):.0f}")

    return records
