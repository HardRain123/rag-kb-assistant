import logging

from fastapi import FastAPI

from app.api.ask import router as ask_router
from app.api.health import router as health_router
from app.api.upload import router as upload_router
from app.core.config import APP_DESCRIPTION, APP_TITLE, APP_VERSION


logging.basicConfig(level=logging.INFO)

# main.py 现在只保留应用入口职责：
# 1. 创建 FastAPI 应用
# 2. 注册已经拆好的业务路由
app = FastAPI(
    title=APP_TITLE,
    version=APP_VERSION,
    description=APP_DESCRIPTION,
)

app.include_router(health_router)
app.include_router(upload_router)
app.include_router(ask_router)
