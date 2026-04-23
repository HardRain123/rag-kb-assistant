from chromadb import chromadb

# ------------------------------
# 2. 初始化 Chroma 客户端
# ------------------------------
# 这里先用最简单的本地客户端
client = None


def get_chroma_client():
    global client
    if client is None:
        client = chromadb.PersistentClient(path="./data/chroma")
    return client


def get_collection(collection_name: str):
    chroma_client = get_chroma_client()
    collection = chroma_client.get_or_create_collection(name=collection_name)
    return collection
