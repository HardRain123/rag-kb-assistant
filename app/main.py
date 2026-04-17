from fastapi import FastAPI, UploadFile
from pydantic import BaseModel
from app.api.chat import router as chat_router
from pathlib import Path
from fastapi import File, Form, HTTPException
from typing import Annotated
import uuid

app = FastAPI(title="RAG KB Assistant", version="0.1.0")

app.include_router(chat_router)


class Item(BaseModel):
    name: str
    description: str | None = None
    tax: float | None = None
    is_offer: bool | None = None


class Message(BaseModel):
    name: str
    price: float
    is_offer: bool | None = None


# 本地存储目录
UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/")
def root():
    return {"message": "RAG KB Assistant is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/docs")
def docs():
    return {"docs": "ok"}


@app.post("/items/")
def create_item(item: Item):
    return {"message": f"Item '{item.name}' with tax {item.tax} created successfully"}


@app.put("/items/{item_id}")
def update_item(item_id: int, item: Item):
    return {
        "message": f"Item with ID {item_id} updated to '{item.name}' with tax {item.tax}"
    }


@app.post("/chat")
def chat(message: Message):
    return {
        "response": f"Received message: '{message.name}' with price {message.price}"
    }


@app.post("/uploadFile")
async def upload_file(file: UploadFile):
    content = await file.read()
    return {"filename": file.filename, "content": content.decode("utf-8")}


@app.post("/uploadFiles")
async def upload_files(
    file: Annotated[UploadFile, File(...)], kb_id: Annotated[str, Form(...)]
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="未选择文件")

    # 生成唯一文件名，避免重名覆盖
    task_id = str(uuid.uuid4())
    save_name = f"{task_id}_{file.filename}"
    save_path = UPLOAD_DIR / save_name

    # 读取上传文件内容
    content = await file.read()

    # 保存到本地
    with open(save_path, "wb") as f:
        f.write(content)

    return {
        "task_id": task_id,
        "filename": file.filename,
        "kb_id": kb_id,
        "saved_path": str(save_path),
    }
