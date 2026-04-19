from pathlib import Path
from typing import Annotated
import uuid

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from chromadb import chromadb

# 导入你刚才写的切片函数
from app.services.chunk_service import split_text


# ------------------------------
# 1. 初始化 FastAPI 应用
# ------------------------------
app = FastAPI(title="RAG KB Assistant", version="0.1.0")

# ------------------------------
# 2. 初始化 Chroma 客户端
# ------------------------------
# 这里先用最简单的本地客户端
client = chromadb.Client()

# ------------------------------
# 3. 本地上传目录
# ------------------------------
UPLOAD_DIR = Path("data/uploads")

# mkdir 的意思是：如果目录不存在，就自动创建
# parents=True 表示上级目录不存在也一起创建
# exist_ok=True 表示目录已存在时不要报错
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------
# 4. 定义请求体模型
# ------------------------------
# Pydantic BaseModel 的作用：
# 用来定义接口接收的数据结构
class IngestRequest(BaseModel):
    saved_path: str
    kb_id: str


class AskRequest(BaseModel):
    # 用户提的问题
    question: str

    # 检索几个 chunk，默认先取 3 个
    top_k: int = 3


# ------------------------------
# 5. 健康检查接口
# ------------------------------
@app.get("/health")
def health():
    """
    最基础的健康检查接口。
    用来确认服务是否正常启动。
    """
    return {"status": "ok"}


# ------------------------------
# 6. 上传 txt 文件接口
# ------------------------------
@app.post("/upload_txt")
async def upload_txt(
    # Annotated[类型, FastAPI参数声明]
    # UploadFile 表示上传文件对象
    file: Annotated[UploadFile, File(...)],
    # kb_id 是一个普通表单字段
    kb_id: Annotated[str, Form(...)],
):
    """
    只负责：
    1. 接收 txt 文件
    2. 保存到本地
    3. 返回保存路径

    暂时不在这里做切片和入库，
    因为一个接口最好职责单一。
    """

    # 如果没选文件，直接报错
    if not file.filename:
        raise HTTPException(status_code=400, detail="未选择文件")

    # 当前阶段只支持 txt
    if not file.filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="当前阶段只支持 txt 文件")

    # 生成一个唯一任务 id
    task_id = str(uuid.uuid4())

    # 保存后的文件名：task_id + 原文件名
    save_name = f"{task_id}_{file.filename}"
    save_path = UPLOAD_DIR / save_name

    # await file.read() 会读取上传文件的原始内容
    # 注意：这里返回的是 bytes（二进制），不是字符串
    content = await file.read()

    # 用 "wb" 二进制方式写文件
    with open(save_path, "wb") as f:
        f.write(content)

    return {
        "task_id": task_id,
        "kb_id": kb_id,
        "filename": file.filename,
        "saved_path": str(save_path),
    }


# ------------------------------
# 7. 入库接口：读取 txt -> 切片 -> 写入 Chroma
# ------------------------------
@app.post("/ingest_txt")
def ingest_txt(req: IngestRequest):
    """
    这个接口负责：
    1. 根据保存路径读取 txt 内容
    2. 调用 split_text 切片
    3. 写入 Chroma
    """

    saved_path = Path(req.saved_path)

    # 如果文件不存在，直接报错
    if not saved_path.exists():
        raise HTTPException(status_code=404, detail="文件不存在，请先上传")

    # 读取 txt 内容
    # encoding="utf-8" 表示按 utf-8 解析文本
    with open(saved_path, "r", encoding="utf-8") as f:
        text = f.read()

    # 调用切片函数
    chunks = split_text(text, chunk_size=500, overlap=100)

    # 如果切完没有内容，说明文件内容为空或切片失败
    if not chunks:
        raise HTTPException(status_code=400, detail="文本为空，无法入库")

    # 文档 id
    # stem 表示文件名去掉扩展名后的部分
    doc_id = saved_path.stem

    # 获取或创建 collection
    # 你可以把它理解成 Chroma 里的一个“集合/表”
    collection = client.get_or_create_collection(name="test_collection")

    # 给每个 chunk 生成唯一 id
    # enumerate(chunks) 会返回：
    # (0, 第一个chunk), (1, 第二个chunk), ...
    chunk_ids = [f"{doc_id}_{idx}" for idx, _ in enumerate(chunks)]

    # 为每个 chunk 准备 metadata
    # metadata 的作用：记录这个 chunk 来自哪里
    metadatas = [
        {
            "chunk_id": chunk_ids[idx],  # 这个 chunk 自己的 id
            "doc_id": doc_id,  # 整个文档的 id
            "kb_id": req.kb_id,  # 所属知识库
            "file_name": saved_path.name,  # 原文件名
            "page": 0,  # txt 暂时没有页码，先写 0
            "chunk_index": idx,  # 第几个 chunk
        }
        for idx, _ in enumerate(chunks)
    ]

    # 写入 Chroma
    # ids / documents / metadatas 三者长度必须一致
    collection.add(
        ids=chunk_ids,
        documents=chunks,
        metadatas=metadatas,
    )

    return {
        "message": "入库成功",
        "doc_id": doc_id,
        "chunk_count": len(chunks),
        "sample_chunk": chunks[0][:100],  # 返回第一个 chunk 的前 100 个字符做示例
    }


# ------------------------------
# 8. 检索接口
# ------------------------------
@app.get("/search")
def search(q: str, n_results: int = 3):
    """
    根据问题 q 去 Chroma 里查最相关的 chunk。
    这里先只做检索，不接大模型。
    """

    collection = client.get_or_create_collection(name="test_collection")

    # query_texts 是“查询文本”
    # Chroma 会自动把它转成向量，再去做相似搜索
    results = collection.query(query_texts=[q], n_results=n_results)

    return results


@app.post("/ask")
def ask(req: AskRequest):
    """
    最小问答接口

    步骤：
    1. 用户提问
    2. 先去 Chroma 检索最相关的 chunks
    3. 把这些 chunks 当作上下文
    4. 返回 answer + snippets + sources
    """

    # 获取 collection
    collection = client.get_or_create_collection(name="test_collection")

    # 用用户问题去做相似检索
    results = collection.query(query_texts=[req.question], n_results=req.top_k)

    # Chroma 返回的是二维结构
    # 因为它支持一次查多个 query
    # 我们现在只查 1 个 query，所以取第 0 个
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    # 如果没查到内容
    if not documents:
        return {
            "question": req.question,
            "answer": "没有检索到相关内容，暂时无法回答。",
            "snippets": [],
            "sources": [],
            "distances": [],
        }

    # 先用假回答函数生成 answer
    answer = build_fake_answer(req.question, documents)

    return {
        "question": req.question,
        "answer": answer,
        "snippets": documents,  # 检索到的原文片段
        "sources": metadatas,  # 每个片段的来源信息
        "distances": distances,  # 相似度距离
    }


def build_fake_answer(question: str, snippets: list[str]) -> str:
    """
    先用一个假的回答函数，帮助你理解 ask 的完整流程。
    后面你再把这里替换成真实模型调用。
    """

    if not snippets:
        return "没有检索到相关内容，暂时无法回答。"

    # 这里先简单返回“基于检索内容的摘要”
    # 后面再替换成真实大模型调用
    joined = "\n".join(snippets[:2])

    return f"问题：{question}\n\n基于检索到的内容，相关信息如下：\n{joined[:300]}"
