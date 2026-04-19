from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

_default_ef = DefaultEmbeddingFunction()


def embed_text(text: str):
    return _default_ef([text])[0]


def embed_texts(texts: list[str]):
    return _default_ef(texts)
