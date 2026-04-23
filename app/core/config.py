
from pathlib import Path


# 应用入口和多个路由都会用到的共享配置，集中放在这里，
# 这样后面想改默认知识库或上传目录时，不需要回 main.py 里找常量。
APP_TITLE = "RAG KB Assistant"
APP_VERSION = "0.1.0"
APP_DESCRIPTION = "面向理赔场景的知识库问答系统"

# Day 9 之后默认走实验用的 B 库。
DEFAULT_COLLECTION_NAME = "claim_kb_b"
DEFAULT_FALLBACK_DISTANCE = 10**9

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
