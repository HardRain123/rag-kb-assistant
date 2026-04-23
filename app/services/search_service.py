import logging
from typing import Any

from app.core.chroma_client import get_collection
from app.core.config import DEFAULT_COLLECTION_NAME, DEFAULT_FALLBACK_DISTANCE
from app.services.rewrite_service import build_rewrite_hints, rewrite_query
from app.services.rerank_service import rerank_result


def query_collection(
    collection_name: str, query_text: str, n_results: int, kb_id: str | None = None
) -> dict[str, Any]:
    """统一封装 Chroma 查询，避免 /search 和 /ask 各自维护一套逻辑。"""
    collection = get_collection(collection_name=collection_name)
    query_kwargs: dict[str, Any] = {
        "query_texts": [query_text],
        "n_results": max(1, n_results),
    }

    if kb_id:
        query_kwargs["where"] = {"kb_id": kb_id}

    return collection.query(**query_kwargs)


def extract_result_items(results: dict[str, Any]) -> list[dict[str, Any]]:
    """
    把 Chroma 的二维返回结构摊平成统一 item 列表，
    这样后面截断、合并、排序都会更直观。
    """
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
    Day 9 的关键点之一：
    rewrite 不覆盖 baseline，而是把 raw / rewrite 两路结果合并。
    """
    merged: dict[str, dict[str, Any]] = {}

    for item in extract_result_items(raw_results) + extract_result_items(
        rewrite_results
    ):
        source = item["source"]
        key = source.get("chunk_id") or f"{source.get('doc_id')}::{item['snippet']}"
        existing = merged.get(key)
        if existing is None or item["distance"] < existing["distance"]:
            merged[key] = item

    return limit_result_items(list(merged.values()), limit)


def build_search_payload(
    query_text: str,
    items: list[dict[str, Any]],
    rewritten_query: str | None,
    used_queries: list[str],
    rewrite_hints: list[str],
) -> dict[str, Any]:
    return {
        "query": query_text,
        "rewritten_query": rewritten_query,
        "used_queries": used_queries,
        "rewrite_hints": rewrite_hints,
        "snippets": [item["snippet"] for item in items],
        "sources": [item["source"] for item in items],
        "distances": [item["distance"] for item in items],
    }


def search_with_optional_rewrite(
    query_text: str,
    n_results: int,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    kb_id: str | None = None,
    use_rewrite: bool = False,
) -> dict[str, Any]:
    """
    /search 和 /ask 共用的主检索链路：
    先 raw recall，再可选 rewrite，最后统一整理返回结构。
    """
    recall_n = max(n_results, 4)

    raw_results = query_collection(
        collection_name=collection_name,
        query_text=query_text,
        n_results=recall_n,
        kb_id=kb_id,
    )
    logging.info("Raw recall results: %s", raw_results)
    rerank_result(query_text, raw_results)
    logging.info("Reranked raw recall results: %s", raw_results)

    rewritten_query = None
    used_queries = [query_text]
    rewrite_hints: list[str] = []

    if not use_rewrite:
        raw_items = limit_result_items(extract_result_items(raw_results), n_results)
        logging.info("Final search results (raw only): %s", raw_items)
        return build_search_payload(
            query_text=query_text,
            items=raw_items,
            rewritten_query=rewritten_query,
            used_queries=used_queries,
            rewrite_hints=rewrite_hints,
        )

    rewrite_hints = build_rewrite_hints(raw_results)
    try:
        rewritten_query = rewrite_query(query_text, rewrite_hints)
    except Exception as exc:
        logging.warning("Query rewrite failed and will fall back to raw query: %s", exc)
        rewritten_query = None

    if not rewritten_query:
        raw_items = limit_result_items(extract_result_items(raw_results), n_results)
        return build_search_payload(
            query_text=query_text,
            items=raw_items,
            rewritten_query=None,
            used_queries=used_queries,
            rewrite_hints=rewrite_hints,
        )

    rewrite_results = query_collection(
        collection_name=collection_name,
        query_text=rewritten_query,
        n_results=recall_n,
        kb_id=kb_id,
    )

    merged_items = merge_query_results(raw_results, rewrite_results, n_results)
    used_queries.append(rewritten_query)
    logging.info("Rewrite query used for retrieval: %s", rewritten_query)

    return build_search_payload(
        query_text=query_text,
        items=merged_items,
        rewritten_query=rewritten_query,
        used_queries=used_queries,
        rewrite_hints=rewrite_hints,
    )
