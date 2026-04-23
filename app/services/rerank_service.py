import re
from typing import Any
import jieba


TITLE_PATTERN = re.compile(r"^(第?\d+[章节]|小结|总结|结论|要点|#{1,6}\s*.+)")
# 1. 停用词：去掉“问法词 / 语气词 / 指代词”，保留真正有业务意义的词
STOP_WORDS = {
    # 问法词
    "怎么",
    "如何",
    "怎样",
    "咋",
    "怎么办",
    "怎么处理",
    "如何处理",
    "什么",
    "哪些",
    "哪个",
    "哪种",
    "哪类",
    "多少",
    # 咨询/口语词
    "请问",
    "想问",
    "我想问",
    "咨询",
    "了解",
    "说下",
    "说说",
    "讲下",
    "讲讲",
    "看下",
    "看一看",
    "看看",
    "告诉我",
    "帮我看下",
    "帮我看看",
    "帮我查下",
    "帮忙看下",
    # 语气词
    "一下",
    "下",
    "呢",
    "吗",
    "嘛",
    "吧",
    "呀",
    "啊",
    "哈",
    # 指代词
    "这个",
    "这个问题",
    "这个情况",
    "这种",
    "这种情况",
    "这类",
    "那个",
    "那种",
    "那类",
    # 弱语义功能词
    "有关",
    "关于",
    "对于",
    "相关",
    "需要",
    "是否需要",
    "一般",
    "通常",
    "可以",
    "能",
    "能够",
    "应该",
    "是不是",
    "有没有",
    "有吗",
}

# 2. 保护词：这些词即使很短，也不要过滤
PROTECTED_TERMS = {
    "理赔",
    "报案",
    "申请",
    "补件",
    "审核",
    "结案",
    "到账",
    "材料",
    "发票",
    "病历",
    "住院",
    "门诊",
    "医院",
    "拒赔",
    "免责",
    "等待期",
    "诊断证明",
    "费用明细",
    "出院小结",
    "非约定医院",
    # 否定/边界词，千万别当 stopword
    "不",
    "未",
    "无",
    "非",
    "没",
    "不能",
    "是否",
}

DOMAIN_TERMS = {
    "理赔申请",
    "非约定医院",
    "费用明细",
    "出院小结",
    "诊断证明",
    "等待期",
    "免责",
    "补件",
}


def rerank_result(question: str, results: dict[str, Any]) -> dict[str, Any]:
    """
    Re-rank one Chroma result batch in place while preserving the original
    two-dimensional response shape:

    - documents: [[...]]
    - metadatas: [[...]]
    - distances: [[...]]
    """

    documents = results.get("documents", [[]]) or [[]]
    metadatas = results.get("metadatas", [[]]) or [[]]
    distances = results.get("distances", [[]]) or [[]]

    snippets = documents[0] if documents else []
    metadata_list = metadatas[0] if metadatas else []
    distance_list = distances[0] if distances else []
    init_tokenizer()

    question_words = [
        word for word in jieba.lcut(question) if word not in STOP_WORDS and word.strip()
    ]
    question_words = filter_query_tokens(question_words)

    scored: list[tuple[float, int, str, dict[str, Any], float | None]] = []
    for idx, doc in enumerate(snippets):
        metadata = metadata_list[idx] if idx < len(metadata_list) else {}
        distance = distance_list[idx] if idx < len(distance_list) else None

        keyword_hits = sum(1 for word in question_words if word in doc)
        title_bonus = 2 if TITLE_PATTERN.search(doc) else 0
        distance_score = -float(distance) if distance is not None else 0.0

        score = keyword_hits * 3 + title_bonus + distance_score
        scored.append((score, idx, doc, metadata, distance))

    scored.sort(key=lambda item: item[0], reverse=True)

    reordered_docs = [doc for _, _, doc, _, _ in scored]
    reordered_metadatas = [metadata for _, _, _, metadata, _ in scored]
    reordered_distances = [distance for _, _, _, _, distance in scored]

    results["documents"] = [reordered_docs]
    results["metadatas"] = [reordered_metadatas]
    results["distances"] = [reordered_distances]
    return results


def cut_text_by_tokens(text: str, limit: int) -> str:
    tokens = list(jieba.cut(text))
    if len(tokens) <= limit:
        return text
    return "".join(tokens[:limit])


def filter_query_tokens(tokens: list[str]) -> list[str]:
    filtered = []
    for token in tokens:
        token = token.strip()
        if not token:
            continue
        if token in PROTECTED_TERMS:
            filtered.append(token)
            continue
        if token in STOP_WORDS:
            continue
        if len(token) < 2:
            continue
        filtered.append(token)
    return filtered


def init_tokenizer():
    for term in DOMAIN_TERMS:
        jieba.add_word(term)
