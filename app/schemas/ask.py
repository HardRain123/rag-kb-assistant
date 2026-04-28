
from pydantic import BaseModel


class AskRequest(BaseModel):
    question: str
    top_k: int = 3
    use_rewrite: bool = False


class CitationItem(BaseModel):
    doc_id: str
    file_name: str
    chunk_id: str
    snippet: str
    doc_type: str | None = None
    chunk_index: int | None = None
    distance: float | None = None


class AskResponse(BaseModel):
    request_id: str
    question: str
    answer: str
    snippets: list[str]
    sources: list[dict]
    distances: list[float]
    rewritten_query: str | None = None
    used_queries: list[str]
    rewrite_hints: list[str]
    fallback_reason: str | None = None
    fallback_reasons: list[str]
    citations: list[CitationItem]
    confidence: str
