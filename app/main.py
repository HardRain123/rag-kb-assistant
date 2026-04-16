from fastapi import FastAPI

app = FastAPI(title="RAG KB Assistant", version="0.1.0")


@app.get("/")
def root():
    return {"message": "RAG KB Assistant is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/docs")
def docs():
    return {"docs": "ok"}
