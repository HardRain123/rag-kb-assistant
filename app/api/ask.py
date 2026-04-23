from fastapi import APIRouter

from app.core.config import DEFAULT_COLLECTION_NAME
from app.core.llm_client import call_llm
from app.schemas.ask import AskRequest
from app.services.search_service import search_with_optional_rewrite


router = APIRouter(tags=["ask"])


def build_prompt(question: str, snippets: list[str]) -> str:
    """
    先把检索片段拼成上下文，再交给大模型回答。
    prompt 继续保持“找不到就明确说没有”的保守策略。
    """
    context = "\n\n".join(snippets)

    prompt = f"""
    你是一个理赔知识库问答助手。
    请严格根据【上下文】回答【问题】。
    如果上下文中找不到答案，就明确回答：未在文档中找到答案。
    不要编造，不要补充上下文以外的信息。

    【问题】
    {question}

    【上下文】
    {context}
    """.strip()

    return prompt.strip()


@router.get("/search")
def search(
    q: str,
    kb_id: str | None = None,
    n_results: int = 3,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    use_rewrite: bool = False,
):
    return search_with_optional_rewrite(
        query_text=q,
        n_results=n_results,
        collection_name=collection_name,
        kb_id=kb_id,
        use_rewrite=use_rewrite,
    )


@router.post("/ask")
def ask(req: AskRequest):
    search_result = search_with_optional_rewrite(
        query_text=req.question,
        n_results=req.top_k,
        collection_name=DEFAULT_COLLECTION_NAME,
        use_rewrite=req.use_rewrite,
    )
    documents = search_result["snippets"]

    if not documents:
        return {
            "question": req.question,
            "answer": "没有检索到相关内容，暂时无法回答。",
            "snippets": [],
            "sources": [],
            "distances": [],
            "rewritten_query": search_result["rewritten_query"],
            "used_queries": search_result["used_queries"],
            "rewrite_hints": search_result["rewrite_hints"],
        }

    prompt = build_prompt(req.question, documents)
    answer = call_llm(prompt)

    return {
        "question": req.question,
        "answer": answer,
        "snippets": documents,
        "sources": search_result["sources"],
        "distances": search_result["distances"],
        "rewritten_query": search_result["rewritten_query"],
        "used_queries": search_result["used_queries"],
        "rewrite_hints": search_result["rewrite_hints"],
    }
