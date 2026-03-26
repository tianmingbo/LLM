import os
import uuid
import json
from typing import AsyncIterator

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from app.memory import close_redis_client, get_redis_history, to_api_messages
from app.roles import ROLES, list_roles
from app.schemas import ChatRequest, ChatResponse, SessionHistoryResponse, SessionSummaryResponse

load_dotenv()

app = FastAPI(title="XingYu Assistant API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SYSTEM_TEMPLATE = """
你是星语助手，一个多角色聊天机器人。
当前激活角色设定如下：
{role_prompt}

请严格遵循该角色风格回答。
如果用户提问不清晰，先澄清再回答。
""".strip()

TITLE_SUMMARY_TEMPLATE = """
请根据下面会话内容，生成一个简短中文会话标题。
要求：
1) 8-16个字
2) 不要标点符号
3) 只输出标题本身，不要解释
4) 能概括本次会话核心主题

会话内容：
{conversation}
""".strip()


def get_llm(streaming: bool = False) -> ChatOpenAI:
    dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")
    if not dashscope_api_key:
        raise HTTPException(status_code=500, detail="DASHSCOPE_API_KEY not set")

    model_name = os.getenv("QWEN_MODEL", "qwen-plus")
    base_url = os.getenv(
        "DASHSCOPE_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    return ChatOpenAI(
        model=model_name,
        api_key=dashscope_api_key,
        base_url=base_url,
        temperature=0.3,
        streaming=streaming,
    )


def build_conversation_text(messages: list, limit: int = 12) -> str:
    if not messages:
        return ""

    parts: list[str] = []
    for msg in messages[-limit:]:
        msg_type = getattr(msg, "type", "")
        role = "用户" if msg_type == "human" else "助手" if msg_type == "ai" else "系统"
        content = str(getattr(msg, "content", "")).strip()
        if not content:
            continue
        parts.append(f"{role}: {content}")
    return "\n".join(parts)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/roles")
async def get_roles() -> dict:
    return {"roles": list_roles()}


@app.get("/sessions/{session_id}", response_model=SessionHistoryResponse)
async def get_session_history(session_id: str) -> SessionHistoryResponse:
    history = get_redis_history(session_id)
    messages = await history.get_messages()
    return SessionHistoryResponse(session_id=session_id, messages=to_api_messages(messages))


@app.delete("/sessions/{session_id}")
async def clear_session(session_id: str) -> dict:
    history = get_redis_history(session_id)
    await history.clear()
    return {"ok": True, "session_id": session_id}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    if req.role not in ROLES:
        raise HTTPException(status_code=400, detail="invalid role")

    session_id = req.session_id or str(uuid.uuid4())
    role = ROLES[req.role]
    llm = get_llm(streaming=False)

    history = get_redis_history(session_id)
    history_messages = await history.get_messages()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_TEMPLATE),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{user_input}"),
        ]
    )

    chain = prompt | llm | StrOutputParser()
    answer = await chain.ainvoke(
        {
            "role_prompt": role.prompt,
            "history": history_messages,
            "user_input": req.message,
        }
    )

    await history.add_user_message(req.message)
    await history.add_ai_message(answer)

    return ChatResponse(session_id=session_id, role=role.key, answer=answer)


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest) -> StreamingResponse:
    if req.role not in ROLES:
        raise HTTPException(status_code=400, detail="invalid role")

    session_id = req.session_id or str(uuid.uuid4())
    role = ROLES[req.role]
    llm = get_llm(streaming=True)

    history = get_redis_history(session_id)
    history_messages = await history.get_messages()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_TEMPLATE),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{user_input}"),
        ]
    )
    chain = prompt | llm | StrOutputParser()

    async def event_stream() -> AsyncIterator[str]:
        answer_chunks: list[str] = []
        try:
            async for chunk in chain.astream(
                {
                    "role_prompt": role.prompt,
                    "history": history_messages,
                    "user_input": req.message,
                }
            ):
                if not chunk:
                    continue
                answer_chunks.append(chunk)
                yield f"data: {json.dumps({'type': 'token', 'text': chunk}, ensure_ascii=False)}\n\n"

            full_answer = "".join(answer_chunks)
            await history.add_user_message(req.message)
            await history.add_ai_message(full_answer)
            yield f"data: {json.dumps({'type': 'done', 'session_id': session_id, 'role': role.key}, ensure_ascii=False)}\n\n"
        except Exception as exc:  # noqa: BLE001
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/sessions/{session_id}/summary", response_model=SessionSummaryResponse)
async def summarize_session_title(session_id: str) -> SessionSummaryResponse:
    history = get_redis_history(session_id)
    messages = await history.get_messages()
    conversation = build_conversation_text(messages)
    if not conversation:
        return SessionSummaryResponse(session_id=session_id, title="新会话")

    llm = get_llm(streaming=False)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是一个会话标题生成器。"),
            ("human", TITLE_SUMMARY_TEMPLATE),
        ]
    )
    chain = prompt | llm | StrOutputParser()
    title = (await chain.ainvoke({"conversation": conversation})).strip()
    if not title:
        title = "未命名会话"
    return SessionSummaryResponse(session_id=session_id, title=title[:24])


@app.on_event("shutdown")
async def on_shutdown() -> None:
    await close_redis_client()


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
