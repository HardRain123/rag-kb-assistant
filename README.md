# 理赔知识库问答助手（RAG MVP）

一个面向保险理赔资料场景的轻量级 RAG 示例项目，基于 FastAPI 提供文本上传、切片入库、向量检索和基于上下文的问答能力。当前版本重点是把最小可运行链路跑通，适合用于演示、联调和后续功能扩展的基础样板。

## 项目背景

在理赔、报销、补件说明等知识问答场景里，业务规则通常分散在制度说明、材料要求、产品条款和 FAQ 文档中。传统关键词搜索能定位内容，但很难直接给出带来源的自然语言回答。

本项目以 RAG 为核心思路，先把业务文本切成可检索的片段，再把检索结果作为上下文交给大模型生成答案，目标是验证“知识入库 + 相似检索 + 引用回答”这条主链路在理赔场景中的可行性。

当前主流程如下：

```text
TXT 上传 -> 文本切片 -> Chroma 入库 -> 相似检索 -> 大模型生成回答
```

## 业务场景

当前仓库默认面向保险/理赔知识问答，适合以下场景：

- 理赔材料要求查询，例如住院报销需要准备哪些材料
- 报销范围说明查询，例如门诊、住院、发票类费用是否支持报销
- 补件说明辅助，例如材料缺失时需要补交什么内容
- 内部演示或 PoC 验证，用于展示知识库问答最小闭环

仓库内的 `sample_claim.txt` 也围绕理赔申请、初审时效、补件和住院报销资料展开，和当前业务定位一致。

## 技术栈

当前代码中已经直接使用或接入的技术如下：

- Python 3.11
- FastAPI：提供 HTTP API
- Uvicorn：本地开发启动 ASGI 服务
- Pydantic：请求体建模与参数校验
- python-dotenv：从 `.env` 加载模型配置
- OpenAI Python SDK：调用兼容 OpenAI 接口的大模型服务
- ChromaDB：本地持久化向量存储
- 本地文本切片服务：`app/services/chunk_service.py`

说明：

- 当前 README 只描述已经在代码里实际使用的技术，不再把 PDF 解析、异步任务队列、任务状态查询等未落地能力写成现状。
- `app/main.py` 当前运行还依赖 `openai` 和 `chromadb`，但这两项尚未出现在 `requirements.txt` 中，属于当前实现与依赖清单之间的已知差异。

## 项目结构

下面是基于当前仓库的精简结构说明，重点标出真实参与主链路的目录和文件：

```text
rag-kb-assistant/
├── app/
│   ├── main.py
│   ├── services/
│   │   └── chunk_service.py
│   ├── schemas/
│   │   └── chat.py
│   ├── api/
│   │   ├── ask.py
│   │   ├── chat.py
│   │   ├── health.py
│   │   ├── task.py
│   │   └── upload.py
│   ├── core/
│   ├── models/
│   └── utils/
├── data/
│   ├── chroma/
│   └── uploads/
├── .env.example
├── requirements.txt
├── sample_claim.txt
└── README.md
```

结构说明：

- `app/main.py`：当前实际运行入口，核心接口全部定义在这里
- `app/services/chunk_service.py`：文本切片核心逻辑，负责段落切分、句子切分和重叠拼接
- `data/uploads/`：上传后的原始文本文件保存目录
- `data/chroma/`：Chroma 本地持久化数据目录
- `app/api`、`app/core`、`app/models`、`app/utils`：目前主要是预留骨架或占位文件，还不是当前主业务链路的实际承载位置

## 启动方式

### 1. 创建并激活虚拟环境

```bash
python -m venv .venv
```

Windows：

```bash
.venv\Scripts\activate
```

macOS / Linux：

```bash
source .venv/bin/activate
```

### 2. 安装基础依赖

```bash
pip install -r requirements.txt
```

### 3. 补齐当前运行前提

当前代码在 `app/main.py` 中还直接导入并使用了以下库：

- `openai`
- `chromadb`

如果仅安装 `requirements.txt` 后启动报错，请补充安装这两项依赖。

### 4. 配置环境变量

复制 `.env.example` 为 `.env`，并至少填写以下当前代码会实际读取的变量：

```env
MODEL_API_KEY=your_api_key
MODEL_BASE_URL=https://api.openai.com/v1
MODEL_NAME=gpt-4o-mini
```

说明：

- 当前代码实际读取的是 `MODEL_API_KEY`、`MODEL_BASE_URL`、`MODEL_NAME`
- `.env.example` 中的 `EMBEDDING_MODEL`、`VECTOR_DB_PATH`、`UPLOAD_DIR` 目前更像预留示例，当前 `app/main.py` 没有按这些变量驱动路径
- 当前服务依赖 `data/uploads` 和 `data/chroma` 可写

### 5. 启动服务

```bash
uvicorn app.main:app --reload
```

启动后可访问：

- 服务地址：`http://127.0.0.1:8000`
- Swagger 文档：`http://127.0.0.1:8000/docs`

## 核心接口说明

以下接口均来自当前 `app/main.py` 的真实实现。

### `GET /health`

用途：检查服务是否正常启动。

示例响应：

```json
{
  "status": "ok"
}
```

### `POST /upload_txt`

用途：上传 `txt` 文件并保存到本地，返回后续入库所需的 `saved_path`。

请求方式：`multipart/form-data`

关键参数：

- `file`：上传的文本文件，只支持 `.txt`
- `kb_id`：知识库标识，会原样透传到后续入库元数据中

最小请求示例：

```bash
curl -X POST "http://127.0.0.1:8000/upload_txt" \
  -F "file=@sample_claim.txt" \
  -F "kb_id=claim-demo"
```

示例响应：

```json
{
  "task_id": "2bde2ad5-8a9c-48c6-b986-6e8f41465d75",
  "kb_id": "claim-demo",
  "filename": "sample_claim.txt",
  "saved_path": "data/uploads/2bde2ad5-8a9c-48c6-b986-6e8f41465d75_sample_claim.txt"
}
```

### `POST /ingest_txt`

用途：读取已上传的文本文件，完成切片并写入 Chroma。

请求方式：`application/json`

关键参数：

- `saved_path`：`/upload_txt` 返回的本地保存路径
- `kb_id`：所属知识库标识，会写入 chunk metadata

最小请求示例：

```json
{
  "saved_path": "data/uploads/2bde2ad5-8a9c-48c6-b986-6e8f41465d75_sample_claim.txt",
  "kb_id": "claim-demo"
}
```

示例响应：

```json
{
  "message": "入库成功",
  "doc_id": "2bde2ad5-8a9c-48c6-b986-6e8f41465d75_sample_claim",
  "chunk_count": 3,
  "sample_chunk": "理赔申请提交后，一般需要在 3 个工作日内完成初审。"
}
```

### `GET /search`

用途：直接执行相似检索，返回命中的文本片段、metadata 和距离信息。

请求方式：查询参数

关键参数：

- `q`：检索问题或关键词
- `kb_id`：可选，传入时会按 `kb_id` 过滤
- `n_results`：返回结果数量，默认 `3`

最小请求示例：

```bash
curl "http://127.0.0.1:8000/search?q=住院报销需要哪些材料&kb_id=claim-demo&n_results=3"
```

示例响应：

```json
{
  "query": "住院报销需要哪些材料",
  "snippets": [
    "住院医疗报销通常需要提供住院小结、发票、费用清单和身份证明。"
  ],
  "sources": [
    {
      "chunk_id": "2bde2ad5-8a9c-48c6-b986-6e8f41465d75_sample_claim_0",
      "doc_id": "2bde2ad5-8a9c-48c6-b986-6e8f41465d75_sample_claim",
      "kb_id": "claim-demo",
      "file_name": "2bde2ad5-8a9c-48c6-b986-6e8f41465d75_sample_claim.txt",
      "page": 0,
      "chunk_index": 0
    }
  ],
  "distances": [
    0.12
  ]
}
```

### `POST /ask`

用途：先检索相关文本片段，再把片段作为上下文交给大模型生成答案。

请求方式：`application/json`

关键参数：

- `question`：用户问题
- `top_k`：检索片段数量，默认 `3`

最小请求示例：

```json
{
  "question": "住院报销通常需要准备哪些材料？",
  "top_k": 3
}
```

示例响应：

```json
{
  "question": "住院报销通常需要准备哪些材料？",
  "answer": "根据检索到的文档内容，住院医疗报销通常需要提供住院小结、发票、费用清单和身份证明。",
  "snippets": [
    "住院医疗报销通常需要提供住院小结、发票、费用清单和身份证明。"
  ],
  "sources": [
    {
      "chunk_id": "2bde2ad5-8a9c-48c6-b986-6e8f41465d75_sample_claim_0",
      "doc_id": "2bde2ad5-8a9c-48c6-b986-6e8f41465d75_sample_claim",
      "kb_id": "claim-demo",
      "file_name": "2bde2ad5-8a9c-48c6-b986-6e8f41465d75_sample_claim.txt",
      "page": 0,
      "chunk_index": 0
    }
  ],
  "distances": [
    0.12
  ]
}
```

补充说明：

- `/ask` 当前会对整个 `claim_kb` collection 做检索，不支持按 `kb_id` 过滤
- 如果没有检索到内容，会直接返回“没有检索到相关内容，暂时无法回答。”
- 如果模型配置缺失，会返回 500 错误并提示检查 `.env`

## 当前能力边界

为了避免 README 与代码实际能力不一致，当前版本的能力边界明确如下：

- 仅支持 `txt` 文件上传，暂不支持 PDF、Word、扫描件 OCR 等输入
- 上传与入库是两步流程，需要先调用 `/upload_txt`，再调用 `/ingest_txt`
- 向量库 collection 当前固定为 `claim_kb`
- `/search` 支持按 `kb_id` 过滤，但 `/ask` 还不支持按 `kb_id` 隔离检索
- 返回的 `sources` 是原始 metadata 结构，尚未整理成更友好的引用展示格式
- 暂无用户鉴权、权限隔离、任务队列、任务状态查询和前端页面
- 依赖本地 `data/chroma` 持久化目录可正常读写，存储异常会直接影响服务运行
- 当前代码以单文件入口为主，`app/api` 等目录尚未完成真正的模块化拆分

## 说明

- 本次 README 仅对齐当前实现，不代表项目终态设计
- 如果后续补齐 PDF 解析、知识库隔离、前端页面或异步任务机制，README 应再同步更新


# 待解决问题

- 文档被重复切片上传到向量数据库，如何去重？
- 如何做数据清洗，去除文档中冗余的数据？
- 如何选择分片策略达到更好的分片效果？
- 如果向量库被污染，怎么处理？