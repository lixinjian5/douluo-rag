"""
文本清洗模块 —— 负责将原始小说文本转化为干净、结构化的正文。
"""
import re
from .config import KW_AD_PATTERNS

# 无意义空白：行首/行尾空白、多余空行
RE_MULTI_BLANK = re.compile(r"[ \t　]+")           # 空格 / 全角空格
RE_MULTI_NEWLINE = re.compile(r"\n{3,}")               # ≥3 个连续换行 → 2 个
RE_PAGE_SEP = re.compile(r"^-{5,}\s*$", re.MULTILINE)  # 页眉/页脚分割线
RE_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")  # 控制字符（保留 \t \n）


def clean_text(text: str) -> str:
    """
    清洗原始文本：
    1. 移除控制字符
    2. 统一空格为半角空格
    3. 压缩多余空行
    4. 移除分隔线
    """
    text = RE_CONTROL_CHARS.sub("", text)
    text = RE_MULTI_BLANK.sub(" ", text)
    text = RE_MULTI_NEWLINE.sub("\n\n", text)
    text = RE_PAGE_SEP.sub("", text)
    # 去除行首行尾空白（保留段落结构）
    text = "\n".join(line.strip() for line in text.splitlines())
    # 再次压缩可能因 strip 产生的连续空行
    text = RE_MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()


def remove_non_content(text: str) -> str:
    """
    去除文中非正文内容：
    - 作者求票 / 求收藏段落
    - 页脚版权声明
    - 开头序言（如果有明显标记）
    """
    lines = text.splitlines()
    kept = []
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            kept.append(line)
            continue

        # 广告检测：一行命中 ≥ 2 个广告词 → 整行丢弃
        ad_hits = sum(1 for kw in KW_AD_PATTERNS if kw in line_stripped)
        if ad_hits >= 2:
            continue

        kept.append(line)

    return "\n".join(kept).strip()


# 作者后记/求票/新书推广的强特征词
KW_POSTSCRIPT = [
    "新书", "新书上", "书号", "书号是", "冲榜",
    "推荐票", "收藏", "订阅", "月票", "保底",
    "求点击", "求支持", "求推荐", "求收藏",
    "感谢大家", "谢谢大家", "说一声谢谢",
    "陪伴大家", "支持我的", "兄弟姐妹们",
    "不让每一位", "为了六年来", "从未食言",
    "陪伴大家", "陪伴各位",
    "我们的唐门", "唐门在新",
    "下次再见", "下本书", "下个故事",
    "阴阳冕",  # 作者另一本书
]


def trim_author_postscript(text: str) -> str:
    """
    检测并截断文末的作者后记/求票/新书推广等内容。

    策略：从倒数第 5 段开始扫描，如果某段包含 ≥ 2 个后记关键词，
    则将该段及其后所有内容视为后记，予以截断。同时若发现「全书完」
    标记之后有大段的 meta 内容，也一并截除。
    """
    lines = text.splitlines()
    if len(lines) < 10:
        return text

    # 从尾部扫描，找到第一个"强后记信号"段落
    for i in range(len(lines) - 1, max(len(lines) - 30, -1), -1):
        line = lines[i].strip()
        if not line:
            continue
        hits = sum(1 for kw in KW_POSTSCRIPT if kw in line)
        if hits >= 2:
            # 向上找段落边界（空行），确保从完整段落开始截
            cut = i
            while cut > 0 and lines[cut - 1].strip() != "":
                cut -= 1
            return "\n".join(lines[:cut]).strip()

    return text
