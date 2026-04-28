import logging
from typing import Any

from app.core.chroma_client import get_collection
from app.core.config import DEFAULT_COLLECTION_NAME, DEFAULT_FALLBACK_DISTANCE
from app.services.query_intent_service import (
    classify_query_intent,
    get_priority_filters_for_intent,
)
from app.services.rewrite_service import build_rewrite_hints, rewrite_query
from app.services.rerank_service import rerank_result

logger = logging.getLogger(__name__)


def query_collection(
    collection_name: str,
    query_text: str,
    n_results: int,
    kb_id: str | None = None,
    where: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Wrap Chroma querying so /search and /ask share one retrieval path."""
    collection = get_collection(collection_name=collection_name)
    query_kwargs: dict[str, Any] = {
        "query_texts": [query_text],
        "n_results": max(1, n_results),
    }

    merged_where: dict[str, Any] = {}
    if where:
        merged_where.update(where)
    if kb_id:
        merged_where["kb_id"] = kb_id
    if merged_where:
        query_kwargs["where"] = merged_where

    return collection.query(**query_kwargs)


def extract_result_items(results: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten one Chroma query result into a stable item list."""
    documents = results.get("documents", [[]]) or [[]]
    metadatas = results.get("metadatas", [[]]) or [[]]
    distances = results.get("distances", [[]]) or [[]]

    document_list = documents[0] if documents else []
    metadata_list = metadatas[0] if metadatas else []
    distance_list = distances[0] if distances else []

    items: list[dict[str, Any]] = []
    for idx, snippet in enumerate(document_list):
        metadata = {}
        if idx < len(metadata_list) and isinstance(metadata_list[idx], dict):
            metadata = metadata_list[idx]

        distance = float(DEFAULT_FALLBACK_DISTANCE)
        if idx < len(distance_list) and distance_list[idx] is not None:
            try:
                distance = float(distance_list[idx])
            except (TypeError, ValueError):
                distance = float(DEFAULT_FALLBACK_DISTANCE)

        items.append(
            {
                "snippet": snippet,
                "source": metadata,
                "distance": distance,
            }
        )

    return items


def limit_result_items(items: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    return sorted(items, key=lambda item: item["distance"])[:limit]


def merge_query_results(
    raw_results: dict[str, Any], rewrite_results: dict[str, Any], limit: int
) -> list[dict[str, Any]]:
    """
    Deduplicate raw and rewritten results while keeping the existing reranked
    order. If the same chunk shows up twice, keep the lower-distance item.
    """
    merged: dict[str, dict[str, Any]] = {}

    for item in extract_result_items(raw_results) + extract_result_items(
        rewrite_results
    ):
        source = item["source"]
        key = source.get("chunk_id") or f"{source.get('doc_id')}::{item['snippet']}"
        existing = merged.get(key)

        if existing is None:
            merged[key] = item
        elif item["distance"] < existing["distance"]:
            merged[key] = item

    return list(merged.values())[:limit]


def build_results_from_items(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "documents": [[item["snippet"] for item in items]],
        "metadatas": [[item["source"] for item in items]],
        "distances": [[item["distance"] for item in items]],
    }


def merge_result_batches(batches: list[dict[str, Any]]) -> dict[str, Any]:
    merged: dict[str, dict[str, Any]] = {}

    for batch in batches:
        for item in extract_result_items(batch):
            source = item["source"]
            key = source.get("chunk_id") or f"{source.get('doc_id')}::{item['snippet']}"
            existing = merged.get(key)

            if existing is None:
                merged[key] = item
            elif item["distance"] < existing["distance"]:
                merged[key] = item

    return build_results_from_items(list(merged.values()))


def build_priority_query_text(query_text: str, query_intent: str | None) -> str:
    if query_intent == "process_entry":
        return f"{query_text} 报案 通知 申请方式 提交申请"
    if query_intent == "process_channel":
        return f"{query_text} 申请方式 线上申请 线下申请 线上提交 线下提交"
    if query_intent == "process":
        return f"{query_text} 理赔流程 申请步骤 提交申请"
    if query_intent == "material":
        return f"{query_text} 材料清单 发票 费用明细 诊断证明 出院小结"
    if query_intent == "supplement":
        return f"{query_text} 补件 材料不全 审核暂停 费用清单 缺失"
    if query_intent == "condition":
        return f"{query_text} 申请条件 申请资格 谁可以申请 申请人"
    if query_intent == "boundary":
        return f"{query_text} 等待期 免责 医院范围 非约定医院 急诊例外"
    return query_text


def query_with_intent_priority(
    collection_name: str,
    query_text: str,
    n_results: int,
    kb_id: str | None,
    query_intent: str | None,
) -> dict[str, Any]:
    batches = [
        query_collection(
            collection_name=collection_name,
            query_text=query_text,
            n_results=n_results,
            kb_id=kb_id,
        )
    ]

    # 根据问题意图增加一些关键词，提升相关结果的排名。
    priority_filters = get_priority_filters_for_intent(query_intent)
    if not priority_filters:
        return batches[0]

    # Keep one global recall batch, then add a few metadata-filtered batches
    # for the current intent. This gives us intent-aware candidates without
    # depending on hard-coded file names.
    for idx, where in enumerate(priority_filters):
        priority_n = max(2, min(4, n_results))
        priority_query_text = build_priority_query_text(query_text, query_intent)
        if idx == 0:
            priority_n = max(6, n_results)

        batches.append(
            query_collection(
                collection_name=collection_name,
                query_text=priority_query_text,
                n_results=priority_n,
                kb_id=kb_id,
                where=where,
            )
        )

    return merge_result_batches(batches)


def build_search_payload(
    query_text: str,
    items: list[dict[str, Any]],
    rewritten_query: str | None,
    used_queries: list[str],
    rewrite_hints: list[str],
    fallback_reasons: list[str],
) -> dict[str, Any]:
    return {
        "query": query_text,
        "rewritten_query": rewritten_query,
        "used_queries": used_queries,
        "rewrite_hints": rewrite_hints,
        "fallback_reasons": fallback_reasons,
        "fallback_reason": fallback_reasons[-1] if fallback_reasons else None,
        "snippets": [item["snippet"] for item in items],
        "sources": [item["source"] for item in items],
        "distances": [item["distance"] for item in items],
    }


def append_fallback_reason(fallback_reasons: list[str], reason: str) -> None:
    if reason not in fallback_reasons:
        fallback_reasons.append(reason)


def rerank_with_fallback(
    question: str,
    results: dict[str, Any],
    fallback_reasons: list[str],
) -> dict[str, Any]:
    try:
        return rerank_result(question, results)
    except Exception as exc:
        logger.warning("Rerank failed and will fall back to original recall order: %s", exc)
        append_fallback_reason(fallback_reasons, "rerank_failed")
        return results


def search_with_optional_rewrite(
    query_text: str,
    n_results: int,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    kb_id: str | None = None,
    use_rewrite: bool = False,
) -> dict[str, Any]:
    """
    Shared retrieval flow for /search and /ask:
    raw recall first, optional rewrite next, then one final payload shape.
    """
    recall_n = max(n_results, 4)
    query_intent = classify_query_intent(query_text)
    fallback_reasons: list[str] = []

    raw_results = query_with_intent_priority(
        collection_name=collection_name,
        query_text=query_text,
        n_results=recall_n,
        kb_id=kb_id,
        query_intent=query_intent,
    )
    logger.info("Raw recall results: %s", raw_results)

    raw_results = rerank_with_fallback(query_text, raw_results, fallback_reasons)
    logger.info("Reranked raw recall results: %s", raw_results)

    rewritten_query = None
    used_queries = [query_text]
    rewrite_hints: list[str] = []

    if not use_rewrite:
        raw_items = extract_result_items(raw_results)[:n_results]
        logger.info("Final search results (raw only): %s", raw_items)
        return build_search_payload(
            query_text=query_text,
            items=raw_items,
            rewritten_query=rewritten_query,
            used_queries=used_queries,
            rewrite_hints=rewrite_hints,
            fallback_reasons=fallback_reasons,
        )

    rewrite_hints = build_rewrite_hints(raw_results)
    try:
        rewritten_query = rewrite_query(query_text, rewrite_hints)
    except Exception as exc:
        logger.warning("Query rewrite failed and will fall back to raw query: %s", exc)
        append_fallback_reason(fallback_reasons, "rewrite_failed")
        rewritten_query = None

    if not rewritten_query:
        raw_items = extract_result_items(raw_results)[:n_results]
        return build_search_payload(
            query_text=query_text,
            items=raw_items,
            rewritten_query=None,
            used_queries=used_queries,
            rewrite_hints=rewrite_hints,
            fallback_reasons=fallback_reasons,
        )

    rewrite_results = query_with_intent_priority(
        collection_name=collection_name,
        query_text=rewritten_query,
        n_results=recall_n,
        kb_id=kb_id,
        query_intent=query_intent,
    )
    rewrite_results = rerank_with_fallback(
        rewritten_query, rewrite_results, fallback_reasons
    )
    logger.info("Reranked rewrite recall results: %s", rewrite_results)

    merged_items = merge_query_results(raw_results, rewrite_results, n_results)
    used_queries.append(rewritten_query)
    logger.info("Rewrite query used for retrieval: %s", rewritten_query)

    return build_search_payload(
        query_text=query_text,
        items=merged_items,
        rewritten_query=rewritten_query,
        used_queries=used_queries,
        rewrite_hints=rewrite_hints,
        fallback_reasons=fallback_reasons,
    )
