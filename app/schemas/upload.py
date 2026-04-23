
from pydantic import BaseModel


class IngestRequest(BaseModel):
    saved_path: str
    kb_id: str
    strategy: str = "a"
