
import logging
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.core.chroma_client import get_chroma_client
from app.core.config import UPLOAD_DIR
from app.schemas.upload import IngestRequest, ReplaceFileRequest
from app.services.ingest_service import ingest_saved_file, replace_saved_file


router = APIRouter(tags=["upload"])
logger = logging.getLogger(__name__)


@router.post("/upload_file")
async def upload_file(
    file: Annotated[UploadFile, File(...)],
    kb_id: Annotated[str, Form(...)],
):
    """
    这个接口只负责把文件落到本地，
    真正的切片和入库放到 ingest 接口里。
    """
    if not file.filename:
        logger.warning("Upload rejected because no filename was provided")
        raise HTTPException(status_code=400, detail="未选择文件")

    if not file.filename.endswith((".txt", ".docx", ".doc", ".md")):
        logger.warning("Upload rejected because file type is not supported: %s", file.filename)
        raise HTTPException(
            status_code=400, detail="当前阶段只支持 txt、docx、doc 和 md 文件"
        )

    task_id = str(uuid.uuid4())
    save_name = f"{task_id}_{file.filename}"
    save_path = UPLOAD_DIR / save_name

    content = await file.read()
    if not content:
        logger.warning("Upload rejected because file content is empty: %s", file.filename)
        raise HTTPException(status_code=400, detail="上传文件为空")

    with open(save_path, "wb") as output_file:
        output_file.write(content)

    return {
        "task_id": task_id,
        "kb_id": kb_id,
        "filename": file.filename,
        "saved_path": str(save_path),
    }


@router.post("/ingest_file")
def ingest_file(req: IngestRequest):
    saved_path = Path(req.saved_path)
    if not saved_path.exists():
        raise HTTPException(status_code=404, detail="文件不存在，请先上传")

    return ingest_saved_file(
        saved_path=saved_path,
        kb_id=req.kb_id,
        strategy=req.strategy,
        doc_type_override=req.doc_type_override,
    )


@router.post("/replace_file")
def replace_file(req: ReplaceFileRequest):
    saved_path = Path(req.saved_path)
    if not saved_path.exists():
        raise HTTPException(status_code=404, detail="文件不存在，请先上传")

    return replace_saved_file(
        saved_path=saved_path,
        kb_id=req.kb_id,
        strategy=req.strategy,
        old_doc_id=req.old_doc_id,
        new_doc_id=req.new_doc_id,
        doc_type_override=req.doc_type_override,
    )


@router.get("/remove_collection")
def remove_collection(collection_name: str):
    get_chroma_client().delete_collection(name=collection_name)
    return {"message": f"Collection '{collection_name}' 已删除"}
