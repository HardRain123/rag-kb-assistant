import logging
from pathlib import Path
from typing import Annotated, Any
import uuid


from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel


from docx import Document

# 导入你刚才写的切片函数
from app.services.chunk_service import split_text, split_text_v2
from app.core.chroma_client import get_chroma_client, get_collection
from app.core.llm_client import call_llm
from app.services.rewrite_service import build_rewrite_hints, rewrite_query

logging.basicConfig(level=logging.INFO)
# Day 9 之后默认走策略 B 的知识库。
# 原因：你当前的实验、对比文档、以及 rewrite 样例都是基于 claim_kb_b 跑的。
DEFAULT_COLLECTION_NAME = "claim_kb_b"
DEFAULT_FALLBACK_DISTANCE = 10**9
# ------------------------------
# 1. 初始化 FastAPI 应用
# ------------------------------
app = FastAPI(
    title="RAG KB Assistant",
    version="0.1.0",
    description="面向理赔场景的知识库问答系统",
)


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
    strategy: str = "a"  # 默认使用策略 a


class AskRequest(BaseModel):
    # 用户提的问题
    question: str

    # 检索几个 chunk，默认先取 3 个
    top_k: int = 3
    use_rewrite: bool = False  # 是否使用重写功能，默认关闭，便于做 A/B 对比


def built_prompt(question: str, snippets: list[str]) -> str:
    """
    构建给大模型的 prompt

    这里先用一个简单的模板，后面你可以根据需要改得更复杂。
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
# 6. 上传文件接口
# ------------------------------
@app.post("/upload_file")
async def upload_file(
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
    if (
        not file.filename.endswith(".txt")
        and not file.filename.endswith(".docx")
        and not file.filename.endswith(".doc")
        and not file.filename.endswith(".md")
    ):
        raise HTTPException(
            status_code=400, detail="当前阶段只支持 txt、docx、doc 和 md 文件"
        )

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


# @app.post("/upload_docx")
# async def upload_docx(
#     file: Annotated[UploadFile, File(...)],
#     kb_id: Annotated[str, Form(...)],
# ):
#     if not file.filename:
#         raise HTTPException(status_code=400, detail="未选择文件")


#     task_id = str(uuid.uuid4())
#     save_name = f"{task_id}_{file.filename}"
#     save_path = UPLOAD_DIR / save_name

#     content = await file.read()

#     with open(save_path, "wb") as f:
#         f.write(content)

#     return {
#         "task_id": task_id,
#         "kb_id": kb_id,
#         "filename": file.filename,
#         "saved_path": str(save_path),
#     }


# ------------------------------
# 7. 入库接口：读取 文件 -> 切片 -> 写入 Chroma
# ------------------------------
@app.post("/ingest_file")
def ingest_file(req: IngestRequest):
    """
    这个接口负责：
    1. 根据保存路径读取文件内容
    2. 调用 split_text 切片
    3. 写入 Chroma
    """

    saved_path = Path(req.saved_path)

    # 如果文件不存在，直接报错
    if not saved_path.exists():
        raise HTTPException(status_code=404, detail="文件不存在，请先上传")

    text = get_file_context(saved_path)

    # 调用切片函数
    chunks = ingest_with_strategy(
        text, saved_path, req.kb_id, req.strategy, collection_name="claim_kb"
    )
    # chunks = split_text(text, chunk_size=200, overlap=50)

    # 如果切完没有内容，说明文件内容为空或切片失败
    if not chunks:
        raise HTTPException(status_code=400, detail="文本为空，无法入库")

    # 文档 id
    # stem 表示文件名去掉扩展名后的部分
    doc_id = saved_path.stem

    # 获取或创建 collection
    # 你可以把它理解成 Chroma 里的一个“集合/表”
    if req.strategy == "a":
        collection = get_collection(collection_name="claim_kb_a")
    else:
        collection = get_collection(collection_name=DEFAULT_COLLECTION_NAME)

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


def get_file_context(saved_path):
    if saved_path.suffix == ".docx" or saved_path.suffix == ".doc":
        # 读取 docx 内容
        document = Document(saved_path)
        text = "\n".join([para.text for para in document.paragraphs])
    else:
        # 读取 txt 内容
        # encoding="utf-8" 表示按 utf-8 解析文本
        with open(saved_path, "r", encoding="utf-8") as f:
            text = f.read()
    return text


def ingest_with_strategy(
    text: str, saved_path: Path, kb_id: str, strategy: str, collection_name: str
):
    is_markdown = saved_path.suffix.lower() == ".md"

    if strategy == "a":
        chunks = split_text(text, chunk_size=300, overlap=50)
    else:
        chunks = split_text_v2(
            text,
            chunk_size=300,
            overlap=50,
            type="markdown" if is_markdown else "default",
        )
    return chunks


def query_collection(
    collection_name: str, query_text: str, n_results: int, kb_id: str | None = None
) -> dict[str, Any]:
    # 统一封装一层 collection.query。
    # 这样 /search 和 /ask 后面都只走这里，避免两边各写各的查询逻辑。
    collection = get_collection(collection_name=collection_name)
    query_kwargs: dict[str, Any] = {
        "query_texts": [query_text],
        "n_results": max(1, n_results),
    }

    if kb_id:
        query_kwargs["where"] = {"kb_id": kb_id}

    return collection.query(**query_kwargs)


def extract_result_items(results: dict[str, Any]) -> list[dict[str, Any]]:
    # Chroma 的返回是二维结构：
    # documents=[ [...snippets...] ], metadatas=[ [...sources...] ]
    # 这里把它摊平成统一的 item 列表，后面 merge / 截断都更容易处理。
    #
    # results.get("documents", [[]]) 的意思是：
    # - 如果 results 里有 documents，就取它
    # - 如果没有，就给一个默认值 [[]]
    #
    # 后面的 or [[]] 是第二层兜底：
    # - 防止取出来的是 None / [] 这类“空值”
    # - 这样后面统一按二维结构处理，不容易报错
    documents = results.get("documents", [[]]) or [[]]
    metadatas = results.get("metadatas", [[]]) or [[]]
    distances = results.get("distances", [[]]) or [[]]

    # Chroma 支持一次传多个 query。
    # 但我们这里每次只查 1 个 query，所以只取第 0 个。
    #
    # 例如：
    # documents = [[片段1, 片段2, 片段3]]
    # 那 document_list 就是 [片段1, 片段2, 片段3]
    document_list = documents[0] if documents else []
    metadata_list = metadatas[0] if metadatas else []
    distance_list = distances[0] if distances else []

    # 最后我们希望得到的结构是：
    # [
    #   {"snippet": "...", "source": {...}, "distance": 0.12},
    #   {"snippet": "...", "source": {...}, "distance": 0.25},
    # ]
    items: list[dict[str, Any]] = []

    # enumerate(document_list) 会一边遍历 snippet，一边给出当前下标 idx。
    # idx=0 对应第一个 snippet，idx=1 对应第二个 snippet，以此类推。
    for idx, snippet in enumerate(document_list):
        # 先给 metadata 一个空字典默认值。
        # 这样即使后面没取到真实 metadata，也不会因为 metadata.get(...) 报错。
        metadata = {}

        # 这里做两个判断：
        # 1. idx < len(metadata_list)
        #    防止 metadata_list 比 document_list 短，直接取 metadata_list[idx] 会越界报错
        # 2. isinstance(metadata_list[idx], dict)
        #    确保这个位置上拿到的确实是字典，而不是 None、字符串、列表等别的类型
        if idx < len(metadata_list) and isinstance(metadata_list[idx], dict):
            # 只有“下标合法 + 值确实是字典”时，才把它当成当前 snippet 的 metadata。
            metadata = metadata_list[idx]

        # 先给 distance 一个很大的默认值。
        # 这样即使后面没拿到真实距离，这条结果也还能参与排序，只是会排得比较靠后。
        distance = float(DEFAULT_FALLBACK_DISTANCE)

        # 同样先确认下标合法，并且当前位置不是 None。
        if idx < len(distance_list) and distance_list[idx] is not None:
            try:
                # 把距离统一转成 float，后面排序时更稳定。
                distance = float(distance_list[idx])
            except (TypeError, ValueError):
                # 如果这里转换失败，比如拿到了奇怪的值，
                # 就继续保留前面的兜底大数，不让整条链路崩掉。
                distance = float(DEFAULT_FALLBACK_DISTANCE)

        # 把当前 snippet、source、distance 打包成统一结构，追加进 items。
        items.append(
            {
                "snippet": snippet,
                "source": metadata,
                "distance": distance,
            }
        )

    return items


def limit_result_items(items: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    # 按 distance 从小到大排序，然后只取前 limit 条。
    # 因为在 Chroma 这里，distance 越小，通常表示越相关。
    return sorted(items, key=lambda item: item["distance"])[:limit]


def merge_query_results(
    raw_results: dict[str, Any], rewrite_results: dict[str, Any], limit: int
) -> list[dict[str, Any]]:
    # Day 9 的关键点之一：
    # 不要让 rewritten query 直接覆盖 raw query，而是两路都保留，再合并。
    # 这里按 chunk_id 去重；如果同一个 chunk 两边都召回到了，就保留距离更小的那条。
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
    # 统一返回结构，避免 /search 和 /ask 一个字段多、一个字段少。
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
    # 这里是 Day 9 的主链路，/search 和 /ask 共用：
    # 1. 先用 raw query 做一次粗召回
    # 2. 从 raw topN 里抽 hints
    # 3. 可选地生成 rewritten query
    # 4. raw / rewrite 两路结果合并
    # 5. 返回 rewritten_query / used_queries / rewrite_hints
    # Day 9 里建议“先粗召回，再 rewrite”。
    # 所以这里不是直接只查 n_results 条，而是至少先查 4 条，
    # 给 build_rewrite_hints 留一点素材空间。
    recall_n = max(n_results, 4)

    # 第一步：先用原始 query 做一次 raw recall。
    raw_results = query_collection(
        collection_name=collection_name,
        query_text=query_text,
        n_results=recall_n,
        kb_id=kb_id,
    )

    # 先准备几个默认值：
    # - rewritten_query: 默认没有改写
    # - used_queries: 默认只用了原始 query
    # - rewrite_hints: 默认没有 hints
    rewritten_query = None
    used_queries = [query_text]
    rewrite_hints: list[str] = []

    # 如果调用方明确不想用 rewrite，
    # 那就直接把 raw_results 截成前 n_results 条返回。
    if not use_rewrite:
        raw_items = limit_result_items(extract_result_items(raw_results), n_results)
        return build_search_payload(
            query_text=query_text,
            items=raw_items,
            rewritten_query=rewritten_query,
            used_queries=used_queries,
            rewrite_hints=rewrite_hints,
        )

    # 第二步：从 raw recall 结果里抽 hints。
    # 这些 hints 是“受约束 rewrite”的输入，不是给最终答案用的上下文。
    rewrite_hints = build_rewrite_hints(raw_results)
    try:
        # 第三步：基于原问题 + hints 生成 rewritten query。
        rewritten_query = rewrite_query(query_text, rewrite_hints)
    except Exception as exc:
        # rewrite 失败时不能把整个检索链路打断。
        # 这里直接回退到 raw query，保证 baseline 一直可用。
        logging.warning("Query rewrite failed and will fall back to raw query: %s", exc)
        rewritten_query = None

    # 如果没有成功生成 rewritten_query，
    # 就回退到 raw_results，但仍然把 rewrite_hints 返回出去，方便做记录和排查。
    if not rewritten_query:
        raw_items = limit_result_items(extract_result_items(raw_results), n_results)
        return build_search_payload(
            query_text=query_text,
            items=raw_items,
            rewritten_query=None,
            used_queries=used_queries,
            rewrite_hints=rewrite_hints,
        )

    # 第四步：如果改写成功，就再用 rewritten_query 查一次。
    rewrite_results = query_collection(
        collection_name=collection_name,
        query_text=rewritten_query,
        n_results=recall_n,
        kb_id=kb_id,
    )

    # 第五步：把 raw_results 和 rewrite_results 合并。
    # 这样 rewrite 就不会覆盖 baseline，而是与 baseline 并行存在。
    merged_items = merge_query_results(raw_results, rewrite_results, n_results)

    # 把改写后的 query 也记进 used_queries，
    # 后面接口返回时就能明确告诉调用方“这次实际用了哪几个 query”。
    used_queries.append(rewritten_query)
    logging.info("Rewrite query used for retrieval: %s", rewritten_query)

    # 最后统一整理成接口返回结构。
    return build_search_payload(
        query_text=query_text,
        items=merged_items,
        rewritten_query=rewritten_query,
        used_queries=used_queries,
        rewrite_hints=rewrite_hints,
    )


# ------------------------------
# 8. 检索接口
# ------------------------------
@app.get("/search")
def search(
    q: str,
    kb_id: str | None = None,
    n_results: int = 3,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    use_rewrite: bool = False,
):
    # /search 本身不再单独维护一套 rewrite 逻辑，
    # 直接复用 search_with_optional_rewrite，方便做 A/B 对比。
    return search_with_optional_rewrite(
        query_text=q,
        n_results=n_results,
        collection_name=collection_name,
        kb_id=kb_id,
        use_rewrite=use_rewrite,
    )


@app.get("/remove_collection")
def remove_collection(collection_name: str):
    """
    删除 collection 的接口
    注意：这个操作会永久删除 collection 里的所有数据，请谨慎使用！
    """

    get_chroma_client().delete_collection(name=collection_name)

    return {"message": f"Collection '{collection_name}' 已删除"}


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

    # 问答接口先复用检索接口的主链路，拿到最终 snippets。
    # 这样 rewrite 的行为和返回字段可以与 /search 保持一致。
    search_result = search_with_optional_rewrite(
        query_text=req.question,
        n_results=req.top_k,
        collection_name=DEFAULT_COLLECTION_NAME,
        use_rewrite=req.use_rewrite,
    )
    documents = search_result["snippets"]

    # 如果没查到内容
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

    prompt = built_prompt(req.question, documents)
    answer = call_llm(prompt)

    return {
        "question": req.question,
        "answer": answer,
        "snippets": documents,  # 检索到的原文片段
        "sources": search_result["sources"],  # 每个片段的来源信息
        "distances": search_result["distances"],  # 相似度距离
        "rewritten_query": search_result["rewritten_query"],
        "used_queries": search_result["used_queries"],
        "rewrite_hints": search_result["rewrite_hints"],
    }
