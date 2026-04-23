
from pydantic import BaseModel


class AskRequest(BaseModel):
    question: str
    top_k: int = 3
    use_rewrite: bool = False
