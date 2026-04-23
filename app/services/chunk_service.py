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
    """
    文本切片主入口（策略 B）。

    设计目标：
    - 对结构化文档优先按“标题 -> section”切分，尽量保留语义结构
    - 对 Markdown 单独走专门逻辑，保留父标题路径
    - 对非 Markdown 文档，先按标题切 section，再按长度兜底切分

    参数说明：
    - text: 原始文本
    - chunk_size: 单个 chunk 目标最大长度
    - overlap: 二次切分时相邻 chunk 的重叠长度
    - type: 文档类型，markdown 时走 Markdown 专用逻辑

    返回：
    - list[str]，每个元素是一个最终 chunk
    """

    # Markdown 文档单独处理。
    # 原因：
    # Markdown 标题是天然有层级的（# / ## / ###），
    # 不能简单按普通标题规则切，否则容易丢父级上下文。
    if type == "markdown":
        return split_markdown_sections(text, chunk_size, overlap)

    # 非 Markdown 文档：
    # 先按标题切成多个 section。
    sections = split_by_heading(text, type)

    final_chunks = []

    # 对每个 section 再做长度兜底：
    # - 没超长，直接作为 chunk
    # - 超长，继续用普通文本切片函数 split_text() 细分
    for sec in sections:
        if len(sec) <= chunk_size:
            final_chunks.append(sec)
        else:
            final_chunks.extend(split_text(sec, chunk_size=chunk_size, overlap=overlap))

    # 过滤掉空 chunk
    return [c for c in final_chunks if c.strip()]


def split_markdown_sections(
    text: str, chunk_size: int = 220, overlap: int = 40
) -> list[str]:
    """
    Markdown 专用切片函数。

    设计目标：
    - 按 Markdown 标题层级切分内容
    - 保留“父标题 -> 子标题 -> 正文”的完整路径
    - 避免出现“只有标题、没有正文”的无效 chunk
    - 如果正文太长，只切正文，但每一段都保留同样的标题路径

    举例：
    原文：
        ## 5. 第四步：准备材料
        ### 最小材料包
        通常包括：
        - 身份证件
        - 出院小结

    期望 chunk：
        ## 5. 第四步：准备材料
        ### 最小材料包
        通常包括：
        - 身份证件
        - 出院小结
    """

    # 按行处理 Markdown，因为标题天然就是“按行识别”的
    lines = text.splitlines()

    # 标题栈：
    # 用来保存当前所在的标题路径。
    # 每个元素是一个元组：(标题层级, 标题文本)
    # 例如：
    # [
    #   (2, "## 5. 第四步：准备材料"),
    #   (3, "### 最小材料包")
    # ]
    stack: list[tuple[int, str]] = []

    # body 用来暂存“当前标题路径下的正文”
    body: list[str] = []

    # 最终切片结果
    chunks: list[str] = []

    # Markdown 标题识别正则：
    # - ^(#{1,6})   匹配 1~6 个 #，作为标题层级
    # - \s*         # 后面允许有空格，也允许没空格
    # - (\S.*)?     标题正文，允许为空，但如果有内容必须非空白开头
    heading_re = re.compile(r"^(#{1,6})\s*(\S.*)?$")

    def flush():
        """
        把“当前收集到的正文 + 当前标题路径”结算成一个或多个 chunk。

        为什么需要这个函数：
        - 在扫描 Markdown 过程中，body 只是暂存在内存里
        - 遇到新标题时，需要先把上一块正文保存下来
        - 文件遍历结束时，也要把最后一块正文保存下来

        flush 的核心职责：
        1. 把 body 中的空白行过滤掉，拼成正文字符串
        2. 取出当前标题路径 prefix
        3. prefix + body_text 组成最终 chunk
        4. 如果太长，则只切正文，但每个子 chunk 都带同样的 prefix
        """
        nonlocal body

        # 过滤空白行，再拼接正文。
        # x.strip() 的作用是：
        # - 去掉首尾空白字符
        # - 如果 strip 后为空字符串，说明这一行没有有效内容
        body_text = "\n".join([x for x in body if x.strip()]).strip()

        # 如果当前没有正文，直接清空 body 并返回。
        # 这样可以避免“纯标题单独入库”。
        if not body_text:
            body = []
            return

        # 组装当前标题路径。
        # 例如 stack = [(2, "## A"), (3, "### B")]
        # 则 prefix =
        # ## A
        # ### B
        prefix = "\n".join(title for _, title in stack).strip()

        # 如果没有标题路径，说明这段正文不属于任何已识别标题，
        # 那就把正文本身作为 chunk 处理。
        if not prefix:
            full = body_text
            if len(full) <= chunk_size:
                chunks.append(full)
            else:
                # 如果正文本身太长，继续用普通文本切片逻辑兜底
                chunks.extend(split_text(full, chunk_size=chunk_size, overlap=overlap))
        else:
            # 有标题路径时，需要把 prefix 和正文一起考虑长度。
            # remain 表示“正文最多还能占多少长度”。
            #
            # 为什么要 max(80, ...)：
            # 如果 prefix 很长，chunk_size - len(prefix) 可能太小，
            # 会导致正文被切得过碎，所以这里设一个保底值 80。
            remain = max(80, chunk_size - len(prefix) - 1)

            # 如果“标题路径 + 正文”整体没有超过 chunk_size，
            # 那正文不需要切，直接作为一个 body_chunk。
            #
            # 否则，只切正文 body_text，
            # 每个切出来的小块后面再重新拼上同样的 prefix。
            body_chunks = (
                [body_text]
                if len(prefix) + 1 + len(body_text) <= chunk_size
                else split_text(body_text, chunk_size=remain, overlap=overlap)
            )

            # 给每个正文块补回标题路径，形成最终 chunk
            for bc in body_chunks:
                chunks.append(f"{prefix}\n{bc}".strip())

        # 当前块结算完成后，清空 body，准备收集下一块正文
        body = []

    # 逐行扫描 Markdown
    for line in lines:
        # 先去掉首尾空白，再判断这一行是不是 Markdown 标题
        m = heading_re.match(line.strip())

        if m:
            # 一旦遇到新标题，说明前一个标题块已经结束，
            # 先把前一个块结算掉。
            flush()

            # m.group(1) 取的是正则第 1 组，也就是开头的 # 们
            # 例如：
            # "### 最小材料包" -> m.group(1) == "###"
            # len("###") == 3，所以 level = 3
            level = len(m.group(1))

            # 当前标题的完整文本
            title = line.strip()

            # 更新标题栈 stack：
            # 如果新标题层级 <= 栈顶标题层级，
            # 说明旧的同级/下级标题已经结束了，需要弹出。
            #
            # 例如当前栈是：
            # [(2, "## 准备材料"), (3, "### 最小材料包")]
            #
            # 如果来了一个新的：
            # ### 特殊场景补充包
            #
            # 因为新标题 level=3，旧栈顶也是 3，
            # 所以先把旧的三级标题弹掉，再压入新的三级标题。
            while stack and stack[-1][0] >= level:
                stack.pop()

            # 把当前新标题入栈
            stack.append((level, title))

        else:
            # 如果这一行不是标题，那它就是当前标题路径下的正文
            body.append(line)

    # 循环结束后，还要再 flush 一次。
    #
    # 原因：
    # 循环中的 flush 只会在“遇到下一个标题时”触发，
    # 但最后一个标题块后面已经没有新标题了，
    # 所以必须手动补一次，把最后一块正文保存下来。
    flush()

    # 返回前过滤空 chunk
    return [c for c in chunks if c.strip()]


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

    has_content = False
    for line in lines:
        if is_heading(line, patterns):
            if current and has_content:
                chunks.append("\n".join(current).strip())
                current = []
                has_content = False
            current.append(line)
            continue
        current.append(line)
        has_content = True

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
