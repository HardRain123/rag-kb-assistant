import logging
import time
import uuid

from fastapi import APIRouter, HTTPException, Query

from app.core.config import DEFAULT_COLLECTION_NAME
from app.core.llm_client import call_llm
from app.schemas.ask import AskRequest
from app.services.ask_audit_service import (
    get_ask_audit,
    list_ask_audits,
    safe_save_ask_audit,
    to_json,
)
from app.services.query_intent_service import classify_query_intent
from app.services.search_service import search_with_optional_rewrite


router = APIRouter(tags=["ask"])
logger = logging.getLogger(__name__)


def elapsed_ms(started_at: float) -> int:
    return int((time.perf_counter() - started_at) * 1000)


def extract_fallback_reasons(search_result: dict | None) -> list[str]:
    if not search_result:
        return []

    fallback_reasons = search_result.get("fallback_reasons")
    if isinstance(fallback_reasons, list):
        return [reason for reason in fallback_reasons if reason]

    fallback_reason = search_result.get("fallback_reason")
    return [fallback_reason] if fallback_reason else []


def build_ask_response(
    *,
    request_id: str,
    question: str,
    answer: str,
    search_result: dict | None,
    fallback_reason: str | None = None,
) -> dict:
    search_result = search_result or {}
    fallback_reasons = extract_fallback_reasons(search_result)
    if fallback_reason and fallback_reason not in fallback_reasons:
        fallback_reasons.append(fallback_reason)

    return {
        "request_id": request_id,
        "question": question,
        "answer": answer,
        "snippets": search_result.get("snippets", []) or [],
        "sources": search_result.get("sources", []) or [],
        "distances": search_result.get("distances", []) or [],
        "rewritten_query": search_result.get("rewritten_query"),
        "used_queries": search_result.get("used_queries", []) or [],
        "rewrite_hints": search_result.get("rewrite_hints", []) or [],
        "fallback_reason": fallback_reason or search_result.get("fallback_reason"),
        "fallback_reasons": fallback_reasons,
    }


def build_ask_audit_record(
    *,
    request_id: str,
    req: AskRequest,
    started_at: float,
    status: str,
    answer: str | None,
    search_result: dict | None,
    fallback_reason: str | None = None,
    error: Exception | None = None,
) -> dict:
    search_result = search_result or {}
    snippets = search_result.get("snippets", []) or []
    search_fallback_reason = search_result.get("fallback_reason")

    return {
        "request_id": request_id,
        "status": status,
        "question": req.question,
        "answer": answer,
        "top_k": req.top_k,
        "use_rewrite": req.use_rewrite,
        "rewritten_query": search_result.get("rewritten_query"),
        "used_queries_json": to_json(search_result.get("used_queries", []) or []),
        "retrieval_count": len(snippets),
        "sources_json": to_json(search_result.get("sources", []) or []),
        "distances_json": to_json(search_result.get("distances", []) or []),
        "latency_ms": elapsed_ms(started_at),
        "fallback_reason": fallback_reason or search_fallback_reason,
        "error_type": type(error).__name__ if error else None,
        "error_message": str(error) if error else None,
    }


def build_answer_style_hint(query_intent: str | None) -> str:
    if query_intent in {"process", "process_entry", "process_channel"}:
        return (
            "优先按流程步骤回答，尽量写清“先做什么、再做什么、最后到什么节点”。"
            "如果上下文里提到了线上/线下申请方式，要明确区分。"
        )
    if query_intent == "material":
        return (
            "优先按材料清单回答，尽量区分基础材料、病历材料、费用材料。"
            "如果上下文只覆盖部分材料，要明确说明是“当前上下文能确认的材料”。"
        )
    if query_intent == "supplement":
        return (
            "优先回答“缺什么、怎么补、补完后会怎样”。"
            "如果上下文里有暂停审核、重新提交、复审等信息，尽量按处理步骤组织。"
        )
    if query_intent == "condition":
        return (
            "优先回答“谁可以申请 / 在什么条件下可以申请”。"
            "如果资格信息不完整，要明确说当前上下文未明确申请人资格。"
        )
    if query_intent == "boundary":
        return (
            "优先回答“一般情况是否可赔”，再补充例外场景、补件条件或边界说明。"
            "不要把边界提醒回答成材料清单或流程说明。"
        )
    return "优先直接回答用户真正想知道的核心结论，再补充必要说明。"


def build_prompt(question: str, snippets: list[str], query_intent: str | None) -> str:
    """
    把检索片段拼成上下文后交给大模型回答。
    这里允许模型综合多个 snippet 做部分回答，但仍然不能编造上下文之外的信息。
    """
    context = "\n\n".join(snippets)
    answer_style_hint = build_answer_style_hint(query_intent)

    prompt = f"""
你是一个理赔知识库问答助手。

请严格根据【上下文】回答【问题】，但你可以综合多个片段中的信息进行归纳。

回答规则：
1. 只使用上下文里明确出现的信息，不要编造，不要补充上下文外的常识。
2. 如果多个片段合起来已经足够回答，就整理成一个连贯答案。
3. 如果上下文只能回答一部分，就先回答已经能确认的部分，再明确说明“其余信息在当前上下文中未明确”。
4. 只有在上下文与问题基本不相关，或者完全无法支持核心结论时，才回答：未在文档中找到答案。
5. 回答尽量简洁、直接；如果适合，用 1-3 条要点组织。
6. 回答时优先遵守这条题型指引：{answer_style_hint}

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


@router.get("/ask_audit")
def list_ask_audit_records(
    limit: int = Query(20, ge=1, le=100),
    status: str | None = None,
    fallback_reason: str | None = None,
):
    return {
        "items": list_ask_audits(
            limit=limit,
            status=status or None,
            fallback_reason=fallback_reason or None,
        )
    }


@router.get("/ask_audit/{request_id}")
def get_ask_audit_record(request_id: str):
    record = get_ask_audit(request_id)
    if record is None:
        raise HTTPException(status_code=404, detail="ask audit record not found")
    return record


@router.post("/ask")
def ask(req: AskRequest):
    request_id = str(uuid.uuid4())
    started_at = time.perf_counter()
    search_result = None

    try:
        query_intent = classify_query_intent(req.question)
        search_result = search_with_optional_rewrite(
            query_text=req.question,
            n_results=req.top_k,
            collection_name=DEFAULT_COLLECTION_NAME,
            use_rewrite=req.use_rewrite,
        )
    except Exception as exc:
        answer = "检索过程暂时异常，请稍后再试。"
        logger.warning("Ask search failed and will return fallback response: %s", exc)
        response = build_ask_response(
            request_id=request_id,
            question=req.question,
            answer=answer,
            search_result=None,
            fallback_reason="search_failed",
        )
        safe_save_ask_audit(
            build_ask_audit_record(
                request_id=request_id,
                req=req,
                started_at=started_at,
                status="error",
                answer=answer,
                search_result=search_result,
                fallback_reason="search_failed",
                error=exc,
            )
        )
        return response

    documents = search_result["snippets"]

    if not documents:
        answer = "没有检索到相关内容，暂时无法回答。"
        logger.info("Ask returned no retrieval result for request_id=%s", request_id)
        response = build_ask_response(
            request_id=request_id,
            question=req.question,
            answer=answer,
            search_result=search_result,
            fallback_reason="no_retrieval_result",
        )
        safe_save_ask_audit(
            build_ask_audit_record(
                request_id=request_id,
                req=req,
                started_at=started_at,
                status="no_retrieval",
                answer=answer,
                search_result=search_result,
                fallback_reason="no_retrieval_result",
            )
        )
        return response

    prompt = build_prompt(req.question, documents, query_intent)
    try:
        answer = call_llm(prompt)
    except Exception as exc:
        answer = "模型服务暂时不可用，请稍后再试。"
        logger.warning("Ask LLM failed and will return fallback response: %s", exc)
        response = build_ask_response(
            request_id=request_id,
            question=req.question,
            answer=answer,
            search_result=search_result,
            fallback_reason="llm_failed",
        )
        safe_save_ask_audit(
            build_ask_audit_record(
                request_id=request_id,
                req=req,
                started_at=started_at,
                status="error",
                answer=answer,
                search_result=search_result,
                fallback_reason="llm_failed",
                error=exc,
            )
        )
        return response

    response = build_ask_response(
        request_id=request_id,
        question=req.question,
        answer=answer,
        search_result=search_result,
    )
    safe_save_ask_audit(
        build_ask_audit_record(
            request_id=request_id,
            req=req,
            started_at=started_at,
            status="success",
            answer=answer,
            search_result=search_result,
        )
    )
    return response
