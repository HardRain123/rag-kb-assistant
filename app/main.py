from fastapi import FastAPI
from pydantic import BaseModel
from app.api.chat import router as chat_router

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
