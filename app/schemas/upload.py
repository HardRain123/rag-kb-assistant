
from pydantic import BaseModel


class IngestRequest(BaseModel):
    saved_path: str
    kb_id: str
    strategy: str = "a"
    doc_type_override: str | None = None


class ReplaceFileRequest(IngestRequest):
    old_doc_id: str
    new_doc_id: str | None = None
