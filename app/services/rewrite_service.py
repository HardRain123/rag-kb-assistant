import re
from typing import Any

from ..core.llm_client import call_llm


HEADING_PATTERNS = [
    re.compile(r"^#{1,6}\s*(.+)$"),
    re.compile(r"^第[一二三四五六七八九十百千万0-9]+[章节部分篇]\s*.*$"),
    re.compile(r"^[一二三四五六七八九十]+、.+$"),
    re.compile(r"^（[一二三四五六七八九十0-9]+）.+$"),
    re.compile(r"^\([一二三四五六七八九十0-9]+\).+$"),
    re.compile(r"^[0-9]+[\.、]\s*.+$"),
]

COLLOQUIAL_RULES = [
    # 这一层不是“复杂 NLP”，只是一个很轻的口语补全。
    # 目的：像“谁来报 / 住院要交啥”这种太短的问题，
    # 先补成一个更像检索语句的 seed query，再决定要不要继续交给 LLM。
    (re.compile(r"(谁来报|谁能报|谁可以报)"), "谁可以发起理赔申请或报案"),
    (
        re.compile(r"(住院)?.*(要交啥|要交什么|交啥|交什么)"),
        "住院医疗理赔需要提交哪些材料",
    ),
    (
        re.compile(r"(不够材料|材料不够|材料不齐|材料不全).*(怎么办|咋办|怎么处理)?"),
        "材料不齐时如何补件并继续申请理赔",
    ),
    (
        re.compile(r"这个(标准|规则).*(讲啥|说啥|是什么|讲什么)"),
        "这个规则说明主要讲什么，适用条件和处理标准是什么",
    ),
    (
        re.compile(r"这种情况(怎么办|咋办|怎么处理|咋处理)"),
        "这种情况的标准处理步骤是什么",
    ),
]


def _clean_line(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip(" -\t")


def _compact_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _extract_heading_path(text: str, limit: int = 2) -> str:
    headings: list[str] = []

    for raw_line in text.splitlines():
        line = _clean_line(raw_line)
        if not line:
            continue

        matched_heading = None
        for pattern in HEADING_PATTERNS:
            match = pattern.match(line)
            if match:
                matched_heading = match.group(1) if match.groups() else line
                break

        if matched_heading:
            headings.append(_clean_line(matched_heading))

        if len(headings) >= limit:
            break

    return " > ".join(headings)


def _summarize_snippet(text: str, limit: int = 90) -> str:
    content_lines: list[str] = []

    # 去除标题，拼接正文内容，准备摘要输入
    for raw_line in text.splitlines():
        line = _clean_line(raw_line)
        if not line:
            continue

        if any(pattern.match(line) for pattern in HEADING_PATTERNS):
            continue

        content_lines.append(line)

    # 正文内容清洗，去除多余空格和制表符
    summary = _compact_text(" ".join(content_lines))
    if not summary:
        summary = _compact_text(text)

    if len(summary) > limit:
        return f"{summary[:limit].rstrip()}..."

    return summary


def build_rewrite_hints(results: dict[str, Any], limit: int = 4) -> list[str]:
    # Day 9 里讲的“受约束 rewrite”，关键不是让模型自由发挥，
    # 而是先从 raw recall 结果里抽出几个库内线索，再基于这些线索做改写。
    #
    # 这里每条 hint 只保留三类信息：
    # - 文件名
    # - 标题/章节
    # - 片段摘要
    #
    # 这样既能给模型足够的检索上下文，又能尽量减少它凭空发散。
    # 和 main.py 里一样，这里先把 documents / metadatas 做成稳定的二维结构。
    documents = results.get("documents", [[]]) or [[]]
    metadatas = results.get("metadatas", [[]]) or [[]]

    # 因为这里只处理单 query，所以仍然只取第 0 个。
    document_list = documents[0] if documents else []
    metadata_list = metadatas[0] if metadatas else []

    # hints 是最终要喂给 rewrite prompt 的线索列表。
    hints: list[str] = []

    # seen 用来去重。
    # 因为有时候不同 chunk 拼出来的 hint 文本完全一样，没必要重复塞给模型。
    seen: set[str] = set()

    # idx 是当前 document 的下标，document 是当前召回到的文本片段。
    for idx, document in enumerate(document_list):
        # 如果这个片段本身就是空的，直接跳过。
        if not document or not document.strip():
            continue

        # metadata 默认给空字典，避免后面 metadata.get(...) 报错。
        metadata = {}

        # 这里和 main.py 同一个思路：
        # - 先确认 idx 没越界
        # - 再确认当前位置的值真的是 dict
        if idx < len(metadata_list) and isinstance(metadata_list[idx], dict):
            metadata = metadata_list[idx]

        # file_name 优先取 file_name，没有的话退回 doc_id，再不行就写“未知来源”。
        file_name = metadata.get("file_name") or metadata.get("doc_id") or "未知来源"

        # heading 尝试从 snippet 里抽标题路径。
        heading = _extract_heading_path(document)

        # summary 尝试从 snippet 里抽正文摘要。
        summary = _summarize_snippet(document)

        # 先把文件名放进去，因为这是最基础的来源线索。
        parts = [f"文件：{file_name}"]
        if heading:
            parts.append(f"标题：{heading}")
        if summary:
            parts.append(f"片段：{summary}")

        # 用固定分隔符拼成一条 hint。
        hint = " | ".join(parts)
        if hint in seen:
            continue

        # 记录去重痕迹。
        seen.add(hint)

        # 把这条 hint 真正收进结果。
        hints.append(hint)

        # 最多只取 limit 条，避免 prompt 过长。
        if len(hints) >= limit:
            break

    return hints


def _normalize_rewritten_query(text: str) -> str:
    if not text:
        return ""

    cleaned = text.replace("```", " ").strip()
    cleaned = re.sub(
        r"^(改写后的(?:检索)?问题|重写后的(?:检索)?问题|改写结果|重写结果|检索(?:语句|问题)|查询语句|查询问题|query|rewritten_query)\s*[:：]\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = cleaned.strip("`'\" ")

    lines = [line.strip("`'\" ") for line in cleaned.splitlines() if line.strip()]
    if not lines:
        return ""

    return _compact_text(lines[0])


def _expand_colloquial_question(question: str) -> str:
    # 如果命中了我们手工整理的“极短口语”模式，
    # 先给出一个保守的补全版本，作为 rewrite 的 seed。
    for pattern, replacement in COLLOQUIAL_RULES:
        if pattern.search(question):
            return replacement

    return question


def rewrite_query(question: str, hints: list[str]) -> str | None:
    # 先把原问题做一次基础清洗，比如压缩多余空格。
    original_question = _compact_text(question)
    if not original_question:
        return None

    # 先做一层轻量口语补全，得到 seed_query。
    seed_query = _expand_colloquial_question(original_question)
    # 对非常短的口语问题，优先使用保守的 seed query。
    # 这样可以避免一上来就让 LLM 把 query 改得太“飞”。
    if seed_query != original_question and len(original_question) <= 12:
        return seed_query

    if not hints:
        # 没有 hints 时就不做“受约束 rewrite”了。
        # 只在 seed_query 明显优于原问题时才返回它，否则返回 None 回退 raw query。
        return seed_query if seed_query != original_question else None

    # 把 hints 拼成 prompt 里更容易阅读的列表文本。
    hint_block = "\n".join(f"- {hint}" for hint in hints)

    # 这个 prompt 的核心约束是：
    # - 只改写，不回答
    # - 保留原问题里的业务词
    # - 不允许凭空发明库里没有的新术语
    # - 如果原问题已经足够清楚，就只做最小改写
    prompt = f"""
你要把用户问题改写成更适合知识库检索的查询语句。
不要回答问题，只做改写。
保留原问题里的角色、材料、流程、规则词。
只允许使用原始问题和库内线索里已经出现过的术语，不要凭空新增新术语。
如果原问题已经足够清楚，只做最小改写。
可以参考下面这条口语补全建议，但仍要优先服从库内线索：
{seed_query}
请只输出一行改写后的检索语句，不要附加解释。

原始问题：
{original_question}

库内线索：
{hint_block}
""".strip()

    # 调 LLM 得到改写结果，再做一次输出清洗。
    rewritten = _normalize_rewritten_query(call_llm(prompt))
    if not rewritten or rewritten == original_question:
        # 模型没改出来，或者改写后和原问题等价时，走保守回退。
        return seed_query if seed_query != original_question else None

    # 如果模型真的给出了一个有效的新 query，就返回它。
    return rewritten
