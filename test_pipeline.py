"""
测试脚本 —— 用内置的《斗罗大陆》片段验证分块和分类逻辑。
"""

import json
import sys
import io

# 强制 stdout 使用 utf-8，解决 Windows GBK 终端编码问题
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from douluo_rag.cleaner import clean_text, remove_non_content
from douluo_rag.splitter import extract_chapter_metadata, split_by_chapters, _cn_num_to_int
from douluo_rag.chunker import split_sentences, create_chunks, is_event_anchor
from douluo_rag.classifier import classify_content
from douluo_rag.pipeline import process_text, process_file


# ── 模拟测试文本 ──────────────────────────────────────────────────────
# 构造一个包含多种内容类型的《斗罗大陆》风格片段
TEST_TEXT = r"""
第297章 小舞献祭

唐三呆呆的看着眼前这一幕，身体完全僵硬了。

这是一个完全封闭的空间，周围是一片漆黑的虚无。唯独正中间，悬浮着一块六边形的紫色水晶，散发着令人心悸的光芒。

“哥，我走了。”小舞微微一笑，泪水顺着脸颊滑落。

唐三怒吼一声："不——！"他拼尽全力冲向那道光幕，却被一股无形的力量弹了回来，重重摔在地上。

他翻身跃起，双拳紧握，周身魂力骤然爆发，八蛛矛从背后破体而出，带着尖锐的破空声刺向前方。

三天后，唐三跪在那座新坟前，一动不动。风吹过他的衣角，带来远方的花香。

"小舞，我会带你回家的。"他轻声说，声音沙哑得几乎听不见。

三年后，唐三再次来到这片山谷。桃花依旧盛开，却已物是人非。漫山遍野的粉色花瓣随风飘舞，像是一场无声的哀悼。
"""

# ── 测试用例 ──────────────────────────────────────────────────────────

def test_chapter_metadata():
    """测试章节标题解析"""
    print("=" * 50)
    print("测试 1: 章节标题解析")
    cases = [
        ("第297章 小舞献祭", ("297", "小舞献祭", "")),
        ("第一章 斗罗大陆异界唐三(一)", ("1", "斗罗大陆异界唐三", "一")),
        ("第三十六章 魂尊(三)", ("36", "魂尊", "三")),
        ("第105章 暗器，暗器(续)", ("105", "暗器，暗器(续)", "")),
    ]
    for title, expected in cases:
        result = extract_chapter_metadata(title)
        status = "✓" if result == expected else "✗"
        print(f"  {status} {title} → {result}  (期望: {expected})")


def test_cn_num():
    """测试中文数字转换"""
    print("\n" + "=" * 50)
    print("测试 2: 中文数字转换")
    cases = [
        ("一", 1), ("十", 10), ("十一", 11),
        ("二十", 20), ("三十六", 36), ("九十九", 99),
        ("一百零五", 105), ("三百", 300), ("三百三十六", 336),
        ("一千零一", 1001),
    ]
    for cn, expected in cases:
        result = _cn_num_to_int(cn)
        status = "✓" if result == expected else "✗"
        print(f"  {status} {cn} → {result}  (期望: {expected})")


def test_cleaner():
    """测试文本清洗"""
    print("\n" + "=" * 50)
    print("测试 3: 文本清洗")
    dirty = "唐三  呆呆的  看着。\n\n\n\n\n新的段落。\n-----------\n正文继续。"
    clean = clean_text(dirty)
    print(f"  清洗前长度: {len(dirty)}")
    print(f"  清洗后长度: {len(clean)}")
    print(f"  清洗结果:\n  {repr(clean)}")

    # 广告过滤
    ad_text = "正文内容\n求推荐票，求收藏，谢谢大家支持！\n继续正文"
    result = remove_non_content(ad_text)
    assert "求推荐票" not in result, "广告行未过滤"
    print(f"  广告过滤: ✓")


def test_sentence_split():
    """测试句子切分"""
    print("\n" + "=" * 50)
    print("测试 4: 句子切分")
    sentences = split_sentences(TEST_TEXT)
    print(f"  共 {len(sentences)} 个句子:")
    for i, s in enumerate(sentences[:8]):
        print(f"  [{i}] {s[:60]}...")
    assert len(sentences) > 5, "句子数量异常"
    print(f"  ✓ 切分正常")


def test_event_anchor():
    """测试事件锚点检测"""
    print("\n" + "=" * 50)
    print("测试 5: 事件锚点检测")
    anchors = ["三年后，唐三再次", "次日清晨，阳光", "与此同时，另一边"]
    non_anchors = ["唐三呆呆的", "这是一个", "他翻身跃起"]

    for a in anchors:
        result = is_event_anchor(a)
        status = "✓" if result else "✗"
        print(f"  {status} 锚点: '{a}' → {result}")
    for na in non_anchors:
        result = is_event_anchor(na)
        if result:
            print(f"  ✗ 误判为锚点: '{na}'")


def test_chunking():
    """测试语义分块"""
    print("\n" + "=" * 50)
    print("测试 6: 语义分块")
    # 清理测试文本
    text = clean_text(TEST_TEXT)
    chunks = create_chunks(text, min_chars=100, max_chars=300, overlap_chars=50)
    print(f"  共 {len(chunks)} 个 Chunk:")
    for i, ch in enumerate(chunks):
        print(f"  Chunk {i}: {len(ch)} 字符, 开头: {ch[:50]}...")
    assert 2 <= len(chunks) <= 8, f"分块数量异常: {len(chunks)}"
    print(f"  ✓ 分块正常")


def test_classification():
    """测试内容分类"""
    print("\n" + "=" * 50)
    print("测试 7: 内容分类")

    dialogue = '"哥，我走了。"小舞微微一笑，"我不后悔认识你。"'
    action = "他翻身跃起，双拳紧握，周身魂力骤然爆发，一拳轰向敌人。飞身一脚踢开对方的攻击。"
    environment = "天空湛蓝，微风拂过大地，茂密的森林在阳光下闪烁着翠绿的光芒。远方的山脉巍峨耸立。"

    for label, sample in [("对话", dialogue), ("动作描写", action), ("环境描写", environment)]:
        result = classify_content(sample)
        status = "✓" if result == label else f"(检测为: {result})"
        print(f"  {status} 期望: {label} | {sample[:50]}...")


def test_full_pipeline():
    """测试完整流程"""
    print("\n" + "=" * 50)
    print("测试 8: 完整流程")
    records = process_text(TEST_TEXT)
    print(f"  生成 {len(records)} 条 JSON 记录")

    # 打印前两条
    for rec in records[:2]:
        print(f"\n  --- Chunk: {rec['chunk_id']} ---")
        print(f"  章节: {rec['chapter_title']}")
        print(f"  字数: {rec['char_count']}, 分类: {rec['content_type']}")
        print(f"  文本: {rec['text'][:80]}...")

    # 验证 JSONL 可序列化
    json_str = "\n".join(json.dumps(r, ensure_ascii=False) for r in records)
    _ = json.loads(json_str.split("\n")[0])  # 确保单行可解析
    print(f"\n  ✓ JSONL 格式正常")


# ── 主入口 ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_cn_num()
    test_chapter_metadata()
    test_cleaner()
    test_sentence_split()
    test_event_anchor()
    test_chunking()
    test_classification()
    test_full_pipeline()

    print("\n" + "=" * 50)
    print("所有测试完成！")

    # 如果源文件存在，执行实际处理
    import os
    src = r"D:\project1\douluo\斗罗大陆第一季.txt"
    if os.path.exists(src):
        print(f"\n检测到源文件，执行真实处理...")
        process_file(src)
