# 企业知识库问答系统（RAG）

一个基于 **Python + FastAPI** 构建的企业知识库问答系统示例项目，目标是支持文档上传、文本解析、切片、向量检索与基于引用的问答返回，用于验证大模型在企业知识场景中的落地方式。

---

## 项目目标

本项目聚焦一个最小可运行的 RAG（Retrieval-Augmented Generation，检索增强生成）闭环：

- 支持上传 PDF / TXT 文档
- 提取文档文本并进行切片
- 对文本块生成向量并写入向量库
- 根据用户问题检索相关片段
- 结合检索结果生成答案
- 返回引用来源，提升回答可追溯性

---

## 当前规划

### 第一阶段

- [x] FastAPI 项目初始化
- [x] `/health` 健康检查接口
- [x] `/chat` 基础对话接口
- [ ] `.env` 配置管理
- [ ] 基础日志输出

### 第二阶段

- [ ] `/upload` 文档上传接口
- [ ] PDF/TXT 文本提取
- [ ] 文本切片
- [ ] chunk metadata 管理
- [ ] 向量入库
- [ ] `/tasks/{task_id}` 任务状态查询

### 第三阶段

- [ ] `/ask` 问答接口
- [ ] top-k 相似检索
- [ ] prompt 拼接
- [ ] answer 返回
- [ ] sources / snippet 返回
- [ ] 基础评测集

### 第四阶段

- [ ] README 完整化
- [ ] Dockerfile
- [ ] `.env.example`
- [ ] 架构图
- [ ] Demo 视频

---

## 技术栈

- **Language:** Python 3.10+
- **Web Framework:** FastAPI
- **ASGI Server:** Uvicorn
- **Schema Validation:** Pydantic
- **Config:** python-dotenv
- **Document Parsing:** pypdf
- **Vector Store:** Chroma（可替换）
- **LLM / Embedding:** OpenAI 兼容接口（可替换）

---

## 项目结构

```text
rag-kb-assistant/
├── app/
│   ├── api/
│   │   ├── health.py
│   │   ├── chat.py
│   │   ├── upload.py
│   │   ├── ask.py
│   │   └── tasks.py
│   ├── core/
│   │   ├── config.py
│   │   ├── logger.py
│   │   └── exceptions.py
│   ├── services/
│   │   ├── llm_service.py
│   │   ├── embedding_service.py
│   │   ├── parser_service.py
│   │   ├── chunk_service.py
│   │   ├── vector_service.py
│   │   └── retrieval_service.py
│   ├── models/
│   │   └── task_store.py
│   ├── schemas/
│   │   ├── chat.py
│   │   ├── upload.py
│   │   ├── ask.py
│   │   └── task.py
│   ├── utils/
│   │   ├── file_utils.py
│   │   └── time_utils.py
│   └── main.py
├── data/
│   ├── uploads/
│   ├── parsed/
│   └── vector_store/
├── tests/
├── scripts/
├── .env
├── .env.example
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 快速开始

### 1. 克隆项目

```bash
git clone <your-repo-url>
cd rag-kb-assistant
```

### 2. 创建虚拟环境

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS / Linux:

```bash
source .venv/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 配置环境变量

复制 `.env.example` 为 `.env` 并填写配置。

```bash
cp .env.example .env
```

Windows 可手动复制。

### 5. 启动服务

```bash
uvicorn app.main:app --reload
```

启动后可访问：

- API 服务：`http://127.0.0.1:8000`
- Swagger 文档：`http://127.0.0.1:8000/docs`

---

## 环境变量示例

```env
MODEL_API_KEY=your_api_key
MODEL_BASE_URL=https://api.openai.com/v1
MODEL_NAME=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
VECTOR_DB_PATH=./data/vector_store
UPLOAD_DIR=./data/uploads
```

---

## API 设计

### `GET /health`

健康检查接口。

**示例响应：**

```json
{
  "status": "ok"
}
```

### `POST /chat`

基础对话接口，用于验证模型调用链路。

**示例请求：**

```json
{
  "message": "你好，请介绍一下你自己"
}
```

**示例响应：**

```json
{
  "answer": "你好，我是一个用于验证模型调用链路的示例接口。"
}
```

### `POST /upload`

上传 PDF / TXT 文档并触发解析、切片、向量化流程。

### `GET /tasks/{task_id}`

查询上传任务处理状态。

### `POST /ask`

基于知识库进行问答，并返回引用来源。

**预期响应字段：**

```json
{
  "question": "报销范围包括什么？",
  "answer": "根据文档内容，报销范围包括门诊和住院相关费用。",
  "sources": ["sample.pdf#page=2"],
  "snippets": ["本保障责任包含门诊及住院医疗费用报销。"]
}
```

---

## 核心流程

```text
文档上传
  -> 文本提取
  -> 文本切片
  -> 向量化
  -> 向量入库
  -> 用户提问
  -> 相似检索
  -> 拼接上下文
  -> 模型生成答案
  -> 返回引用
```

---

## 开发路线图

### MVP

- 跑通 `/health`
- 跑通 `/chat`
- 跑通 `/upload`
- 跑通 `/ask`

### 可演示版本

- 支持 PDF / TXT 文档
- 支持 metadata 管理
- 返回引用和 snippet
- 增加日志和错误处理

### 可扩展方向

- 多知识库隔离（`kb_id`）
- 用户鉴权与权限控制
- 异步任务队列
- rerank 优化
- 对接前端页面
- 接入 Spring AI / Java 服务化改造

---

## 测试建议

至少准备以下测试场景：

- 上传 1 个 TXT 文件并完成解析
- 上传 1 个 PDF 文件并完成解析
- 提问命中文档内容
- 返回答案时包含正确引用
- 非法文件类型上传时返回错误
- 空问题请求时返回校验错误

---

## 注意事项

- 当前版本以 MVP 为目标，优先保证主链路跑通
- 不建议在早期阶段引入过多复杂能力（如多智能体、复杂权限、重前端）
- 若 PDF 为扫描件，文本提取效果可能受限
- 向量库与模型服务均可按实际情况替换

---

## 后续计划

- 完成文档上传与解析链路
- 完成向量检索与问答闭环
- 补充评测数据集与日志
- 完成 Docker 化部署
- 增加系统截图与演示视频

---

## License

MIT
