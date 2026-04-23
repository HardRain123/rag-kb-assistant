import os
from fastapi.exceptions import HTTPException
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # 加载 .env 文件中的环境变量

MODEL_API_KEY = os.getenv("MODEL_API_KEY")
MODEL_BASE_URL = os.getenv("MODEL_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")

# 初始化 OpenAI 客户端
llm_client = None


# 获取大模型
def get_model():
    global llm_client
    if llm_client is None:
        llm_client = OpenAI(
            api_key=MODEL_API_KEY,
            base_url=MODEL_BASE_URL,
        )
    return llm_client


def call_llm(prompt: str) -> str:
    if not MODEL_API_KEY or not MODEL_BASE_URL or not MODEL_NAME:
        raise HTTPException(status_code=500, detail="模型配置缺失，请检查 .env")

    response = get_model().chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": "你是一个严格基于上下文回答问题的知识库助手。",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    answer = response.choices[0].message.content
    return answer or "模型未返回有效内容"
