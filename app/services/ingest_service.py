import logging
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from app.core.chroma_client import get_collection
from app.core.config import DEFAULT_COLLECTION_NAME
from app.services.chunk_service import split_text, split_text_v2
from app.services.query_intent_service import (
    extract_heading_path,
    infer_chunk_type,
    infer_doc_type,
)
from app.utils.file_utils import get_file_context

logger = logging.getLogger(__name__)


def ingest_with_strategy(text: str, saved_path: Path, strategy: str) -> list[str]:
    """根据策略选择切片方式。"""
    is_markdown = saved_path.suffix.lower() == ".md"

    if strategy == "a":
        return split_text(text, chunk_size=300, overlap=50)

    return split_text_v2(
        text,
        chunk_size=300,
        overlap=50,
        type="markdown" if is_markdown else "default",
    )


def _build_chunk_metadatas(
    chunk_ids: list[str],
    chunks: list[str],
    doc_id: str,
    kb_id: str,
    file_name: str,
    doc_type_override: str | None = None,
) -> list[dict[str, Any]]:
    doc_type = infer_doc_type(file_name=file_name, doc_type_override=doc_type_override)

    return [
        {
            "chunk_id": chunk_ids[idx],
            "doc_id": doc_id,
            "kb_id": kb_id,
            "file_name": file_name,
            "doc_type": doc_type,
            "chunk_type": infer_chunk_type(
                heading_path=extract_heading_path(chunks[idx]),
                chunk_text=chunks[idx],
            ),
            "heading_path": extract_heading_path(chunks[idx]),
            "page": 0,
            "chunk_index": idx,
        }
        for idx, _ in enumerate(chunk_ids)
    ]


def ingest_saved_file(
    saved_path: Path,
    kb_id: str,
    strategy: str,
    doc_type_override: str | None = None,
    doc_id_override: str | None = None,
) -> dict[str, Any]:
    """
    这里承接“读取文件 -> 切片 -> 写入 Chroma”的主流程，
    路由层只负责接 HTTP 参数和返回结果。
    """
    text = load_file_text(saved_path)
    chunks = ingest_with_strategy(text=text, saved_path=saved_path, strategy=strategy)

    if not chunks:
        raise HTTPException(status_code=400, detail="文本为空，无法入库")

    doc_id = doc_id_override or saved_path.stem
    collection_name = "claim_kb_a" if strategy == "a" else DEFAULT_COLLECTION_NAME
    collection = get_collection(collection_name=collection_name)

    chunk_ids = [f"{doc_id}_{idx}" for idx, _ in enumerate(chunks)]
    metadatas = _build_chunk_metadatas(
        chunk_ids=chunk_ids,
        chunks=chunks,
        doc_id=doc_id,
        kb_id=kb_id,
        file_name=saved_path.name,
        doc_type_override=doc_type_override,
    )

    collection.add(
        ids=chunk_ids,
        documents=chunks,
        metadatas=metadatas,
    )

    return {
        "message": "入库成功",
        "doc_id": doc_id,
        "chunk_count": len(chunks),
        "sample_chunk": chunks[0][:100],
    }


def _build_doc_where(kb_id: str, doc_id: str) -> dict[str, Any]:
    return {"$and": [{"kb_id": kb_id}, {"doc_id": doc_id}]}


def delete_document_chunks(
    kb_id: str,
    doc_id: str,
    strategy: str,
) -> dict[str, Any]:
    collection_name = "claim_kb_a" if strategy == "a" else DEFAULT_COLLECTION_NAME
    collection = get_collection(collection_name=collection_name)
    where = _build_doc_where(kb_id=kb_id, doc_id=doc_id)

    existing = collection.get(where=where)
    deleted_count = len(existing.get("ids", []))

    if deleted_count:
        collection.delete(where=where)

    return {
        "collection_name": collection_name,
        "deleted_doc_id": doc_id,
        "deleted_chunk_count": deleted_count,
    }


def replace_saved_file(
    saved_path: Path,
    kb_id: str,
    strategy: str,
    old_doc_id: str,
    new_doc_id: str | None = None,
    doc_type_override: str | None = None,
) -> dict[str, Any]:
    """
    Replace an already-ingested document with a new saved file.

    The new file is read and chunked before any deletion happens. If the caller
    reuses the old doc_id, delete first to avoid duplicate Chroma ids. Otherwise
    add the new version first, then remove the old version so a failed add does
    not wipe the currently searchable document.
    """
    text = load_file_text(saved_path)
    chunks = ingest_with_strategy(text=text, saved_path=saved_path, strategy=strategy)
    if not chunks:
        raise HTTPException(status_code=400, detail="新文件文本为空，无法替换入库")

    target_doc_id = new_doc_id or saved_path.stem

    if target_doc_id == old_doc_id:
        delete_result = delete_document_chunks(
            kb_id=kb_id,
            doc_id=old_doc_id,
            strategy=strategy,
        )
        ingest_result = ingest_saved_file(
            saved_path=saved_path,
            kb_id=kb_id,
            strategy=strategy,
            doc_type_override=doc_type_override,
            doc_id_override=target_doc_id,
        )
    else:
        ingest_result = ingest_saved_file(
            saved_path=saved_path,
            kb_id=kb_id,
            strategy=strategy,
            doc_type_override=doc_type_override,
            doc_id_override=target_doc_id,
        )
        delete_result = delete_document_chunks(
            kb_id=kb_id,
            doc_id=old_doc_id,
            strategy=strategy,
        )

    return {
        "message": "替换入库成功",
        "old_doc_id": old_doc_id,
        "new_doc_id": ingest_result["doc_id"],
        "chunk_count": ingest_result["chunk_count"],
        "deleted_chunk_count": delete_result["deleted_chunk_count"],
        "collection_name": delete_result["collection_name"],
        "sample_chunk": ingest_result["sample_chunk"],
    }


def load_file_text(saved_path: Path) -> str:
    try:
        text = get_file_context(saved_path)
    except UnicodeDecodeError as exc:
        logger.warning("Ingest rejected because file encoding is not UTF-8: %s", saved_path)
        raise HTTPException(status_code=400, detail="文件编码不是 UTF-8，暂时无法读取") from exc
    except Exception as exc:
        logger.warning("Ingest rejected because file read failed: %s", exc)
        raise HTTPException(status_code=400, detail=f"文件读取失败：{exc}") from exc

    if not text or not text.strip():
        logger.warning("Ingest rejected because extracted text is empty: %s", saved_path)
        raise HTTPException(status_code=400, detail="文本为空，无法入库")

    return text
