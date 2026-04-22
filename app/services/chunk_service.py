import re


def split_text(text: str, chunk_size: int = 250, overlap: int = 50) -> list[str]:
    """
    升级版文本切片函数

    处理顺序：
    1. 先按段落切
    2. 段落太长时，再按句子切
    3. 句子还太长时，最后按固定长度硬切
    4. 相邻 chunk 保留 overlap 个字符重叠

    参数：
    - text: 原始全文字符串
    - chunk_size: 每个 chunk 的目标最大长度
    - overlap: 相邻 chunk 之间保留多少重复字符

    返回：
    - list[str]，每个元素都是一个 chunk 文本
    """

    # 如果文本为空，直接返回空列表
    if not text or not text.strip():
        return []

    # overlap 不能大于等于 chunk_size，否则会死循环
    if overlap >= chunk_size:
        raise ValueError("overlap 必须小于 chunk_size")

    # 统一换行符，避免 Windows / Linux 换行差异
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()

    # 第一步：按段落切
    paragraphs = split_paragraphs(text)

    # 第二步：把段落进一步拆成“适合拼 chunk 的基本单元”
    # 这里的 units 可能是：
    # - 正常段落
    # - 句子
    # - 硬切后的短文本
    units = []

    for para in paragraphs:
        # 如果一个段落本身不长，直接作为一个单元
        if len(para) <= chunk_size:
            units.append(para)
            continue

        # 如果段落太长，就按句子切
        sentences = split_sentences(para)

        # 如果句子切分失败（极少数情况），就直接硬切整个段落
        if not sentences:
            units.extend(force_split(para, chunk_size, overlap=0))
            continue

        for sent in sentences:
            # 正常长度句子，直接加入
            if len(sent) <= chunk_size:
                units.append(sent)
            else:
                # 如果单句仍然超长，再硬切
                units.extend(force_split(sent, chunk_size, overlap=0))

    # 第三步：把这些基本单元重新拼成最终 chunk
    chunks = merge_units(units, chunk_size, overlap)

    return [c for c in chunks if c.strip()]


def split_paragraphs(text: str) -> list[str]:
    """
    按段落切分文本

    规则：
    - 空行优先作为段落边界
    - 普通换行也视作段落边界
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n|\n", text) if p.strip()]
    return paragraphs


def split_sentences(text: str) -> list[str]:
    """
    按中文常见句号切句，保留标点。

    例如：
    '你好。今天天气不错！要出去吗？'
    会切成：
    ['你好。', '今天天气不错！', '要出去吗？']
    """

    # 按标点切分，并保留标点本身
    parts = re.split(r"([。！？；!?;])", text)

    sentences = []
    for i in range(0, len(parts), 2):
        sentence = parts[i].strip()
        punct = parts[i + 1] if i + 1 < len(parts) else ""
        combined = (sentence + punct).strip()
        if combined:
            sentences.append(combined)

    return sentences


def force_split(text: str, chunk_size: int, overlap: int = 0) -> list[str]:
    """
    对超长文本做硬切分。

    什么时候会用到：
    - 一整段没有明显标点
    - 一个句子本身就超长
    """

    if not text.strip():
        return []

    if overlap >= chunk_size:
        raise ValueError("force_split 中 overlap 必须小于 chunk_size")

    result = []
    start = 0
    step = chunk_size - overlap if overlap > 0 else chunk_size

    while start < len(text):
        end = start + chunk_size
        piece = text[start:end].strip()
        if piece:
            result.append(piece)
        start += step

    return result


def merge_units(units: list[str], chunk_size: int, overlap: int) -> list[str]:
    """
    把多个基本单元（段落/句子/硬切结果）拼成最终 chunk。

    规则：
    - 尽量拼到 chunk_size 左右
    - 超过时，先保存当前 chunk
    - 新 chunk 保留前一个 chunk 的尾部 overlap
    """

    chunks = []
    current = ""

    for unit in units:
        # 当前 chunk 为空，直接放进去
        if not current:
            current = unit
            continue

        # 如果拼进去后还没超过限制，就继续拼
        if len(current) + 1 + len(unit) <= chunk_size:
            current += "\n" + unit
        else:
            # 当前块先保存
            chunks.append(current.strip())

            # 取尾部 overlap 字符，减少语义断裂
            tail = current[-overlap:] if overlap > 0 else ""

            # 新块从“尾部重叠 + 当前单元”开始
            current = (tail + "\n" + unit).strip()

            # 理论上 unit 已经不会超长
            # 但为了保险，如果 current 还是超长，继续硬切
            while len(current) > chunk_size:
                chunks.append(current[:chunk_size].strip())
                current = current[chunk_size - overlap :].strip()

    # 循环结束，把最后一个 chunk 补进去
    if current.strip():
        chunks.append(current.strip())

    return chunks


"""
按标题+段落优先的方式切分文本，适合结构化文档（如说明书、报告等），能更好地保持语义完整性。
"""


def split_text_v2(
    text: str, chunk_size: int = 220, overlap: int = 40, type: str = "default"
) -> list[str]:
    # 1. 先按标题切
    sections = split_by_heading(text, type)

    # 2. 对每个 section 再做长度兜底
    final_chunks = []
    for sec in sections:
        if len(sec) <= chunk_size:
            final_chunks.append(sec)
        else:
            final_chunks.extend(split_long_text(sec, chunk_size, overlap))

    return final_chunks


def is_heading(line: str, patterns: str) -> bool:
    line = line.strip()
    if not line:
        return False

    return any(re.match(p, line) for p in patterns)


def split_by_heading(text: str, type: str = "default") -> list[str]:
    lines = text.splitlines()
    chunks = []
    current = []
    if type == "markdown":
        patterns = [r"^#{1,6}\s+.*$"]  # Markdown 标题
    else:
        patterns = [
            r"^第[一二三四五六七八九十百千万0-9]+[章节部分篇]\s*.*$",  # 第1章 / 第一章
            r"^[一二三四五六七八九十]+、.+$",  # 一、二、三、
            r"^（[一二三四五六七八九十0-9]+）.+$",  # （一）（二）（1）
            r"^\([一二三四五六七八九十0-9]+\).+$",  # (一)(二)(1)
            r"^[0-9]+[\.、]\s*.+$",  # 1. xxx / 1、xxx
            r"^[0-9]+\.[0-9]+(\.[0-9]+)*\s+.+$",  # 1.1 xxx / 1.2.3 xxx
        ]  # 常见中文文档标题格式

    for line in lines:
        if is_heading(line, patterns):
            if current:
                chunks.append("\n".join(current).strip())
                current = []
        current.append(line)

    if current:
        chunks.append("\n".join(current).strip())

    return [c for c in chunks if c]


def split_long_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return chunks
