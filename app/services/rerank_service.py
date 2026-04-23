import re

"""
重排序规则：
1、优先匹配问题中的关键词，提升相关度更高的文本片段
2、如果文本片段中包含明显的标题或小结，优先提升这些片段的排名
3、距离问题更近的文本片段优先提升排名
"""


def rerank_result(
    question: str,
    results: dict[str, list[list[str]]],
):
    snippets = results.get("documents", [[]])[0] if results.get("documents") else []
    distances = results.get("distances", [[]])[0] if results.get("distances") else []
    question_words = question.replace("?", " ").replace("？", " ").split(" ")
    scored = []
    for idx, doc in enumerate(snippets):
        keywords_hits = sum(1 for word in question_words if word in doc)
        title_bonus = (
            2
            if re.search(r"^(第?\d+章|第?\d+节|小结|总结|结论|要点|#{1,6}\s*.+)", doc)
            else 0
        )
        disance_score = (
            -distances[idx] if len(distances) > idx else 0
        )  # 距离越近分数越高
        score = keywords_hits * 3 + title_bonus + disance_score
        scored.append((score, idx, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    snippets = [doc for _, _, doc in scored]
    results["documents"] = snippets
