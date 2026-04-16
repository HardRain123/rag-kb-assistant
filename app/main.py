from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="RAG KB Assistant", version="0.1.0")


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
def create_item(item: Message):
    return {
        "message": f"Item '{item.name}' with price {item.price} created successfully"
    }


@app.put("/items/{item_id}")
def update_item(item_id: int, item: Message):
    return {
        "message": f"Item with ID {item_id} updated to '{item.name}' with price {item.price}"
    }


@app.post("/chat")
def chat(message: Message):
    return {
        "response": f"Received message: '{message.name}' with price {message.price}"
    }
