"""
异步流式调用 + API 接口
"""
import os
from typing import AsyncIterator

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_community.chat_models import ChatTongyi
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

load_dotenv(override=True)
qwen_api_key = os.getenv("DASHSCOPE_API_KEY")

if not qwen_api_key:
    raise RuntimeError("缺少环境变量 DASHSCOPE_API_KEY")

model = ChatTongyi(
    model=os.getenv("QWEN_MODEL", "qwen-plus"),
    max_retries=2,
    api_key=qwen_api_key,
    streaming=True,
)

app = FastAPI(title="Qwen Async Streaming API", version="1.0.0")


class ChatRequest(BaseModel):
    user_prompt: str = Field(..., description="用户输入")
    system_prompt: str = Field("你是一个有帮助的助手。", description="系统提示词")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/api/qwen/chat")
async def chat(req: ChatRequest) -> dict:
    messages = [
        SystemMessage(content=req.system_prompt),
        HumanMessage(content=req.user_prompt),
    ]
    response = await model.ainvoke(messages)
    return {"content": response.content}


async def _stream_reply(req: ChatRequest) -> AsyncIterator[str]:
    messages = [
        SystemMessage(content=req.system_prompt),
        HumanMessage(content=req.user_prompt),
    ]
    try:
        async for chunk in model.astream(messages):
            if chunk.content:
                # 使用 SSE 格式，客户端可稳定按块实时消费
                yield f"data: {chunk.content}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as exc:
        yield f"data: [ERROR] {str(exc)}\n\n"


@app.post("/api/qwen/chat/stream")
async def chat_stream(req: ChatRequest) -> StreamingResponse:
    return StreamingResponse(
        _stream_reply(req),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("model.qwen.qwen_async:app", host="0.0.0.0", port=8000, reload=False)
