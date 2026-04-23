from pathlib import Path
from typing import Any

from fastapi import HTTPException

from app.core.chroma_client import get_collection
from app.core.config import DEFAULT_COLLECTION_NAME
from app.services.chunk_service import split_text, split_text_v2
from app.utils.file_utils import get_file_context


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
    chunk_ids: list[str], doc_id: str, kb_id: str, file_name: str
) -> list[dict[str, Any]]:
    return [
        {
            "chunk_id": chunk_ids[idx],
            "doc_id": doc_id,
            "kb_id": kb_id,
            "file_name": file_name,
            "page": 0,
            "chunk_index": idx,
        }
        for idx, _ in enumerate(chunk_ids)
    ]


def ingest_saved_file(saved_path: Path, kb_id: str, strategy: str) -> dict[str, Any]:
    """
    这里承接“读取文件 -> 切片 -> 写入 Chroma”的主流程，
    路由层只负责接 HTTP 参数和返回结果。
    """
    text = get_file_context(saved_path)
    chunks = ingest_with_strategy(text=text, saved_path=saved_path, strategy=strategy)

    if not chunks:
        raise HTTPException(status_code=400, detail="文本为空，无法入库")

    doc_id = saved_path.stem
    collection_name = "claim_kb_a" if strategy == "a" else DEFAULT_COLLECTION_NAME
    collection = get_collection(collection_name=collection_name)

    chunk_ids = [f"{doc_id}_{idx}" for idx, _ in enumerate(chunks)]
    metadatas = _build_chunk_metadatas(
        chunk_ids=chunk_ids,
        doc_id=doc_id,
        kb_id=kb_id,
        file_name=saved_path.name,
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
