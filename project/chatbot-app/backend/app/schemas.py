from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    role: str = Field(..., min_length=1, max_length=64)
    session_id: str | None = Field(default=None, max_length=128)


class ChatResponse(BaseModel):
    session_id: str
    role: str
    answer: str


class MessageItem(BaseModel):
    role: str
    content: str


class SessionHistoryResponse(BaseModel):
    session_id: str
    messages: list[MessageItem]


class SessionSummaryResponse(BaseModel):
    session_id: str
    title: str
