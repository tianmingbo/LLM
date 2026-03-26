import json
import os

import redis.asyncio as redis
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, messages_from_dict, messages_to_dict

_redis_client: redis.Redis | None = None


def _get_redis_client() -> redis.Redis:
    global _redis_client
    if _redis_client is None:
        redis_url = os.getenv("REDIS_URL", "redis://:tian666@localhost:6389/0")
        _redis_client = redis.from_url(redis_url, decode_responses=True)
    return _redis_client


async def close_redis_client() -> None:
    global _redis_client
    if _redis_client is not None:
        await _redis_client.aclose()
        _redis_client = None


class AsyncRedisChatMessageHistory:
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.client = _get_redis_client()
        self.key_prefix = os.getenv("MEMORY_KEY_PREFIX", "chat:")
        self.ttl_seconds = int(os.getenv("MEMORY_TTL_SECONDS", "86400"))

    @property
    def key(self) -> str:
        return f"{self.key_prefix}{self.session_id}"

    async def get_messages(self) -> list[BaseMessage]:
        raw_items = await self.client.lrange(self.key, 0, -1)
        if not raw_items:
            return []
        payload = [json.loads(item) for item in raw_items]
        return messages_from_dict(payload)

    async def add_messages(self, messages: list[BaseMessage]) -> None:
        if not messages:
            return
        serialized = [json.dumps(item, ensure_ascii=False) for item in messages_to_dict(messages)]
        await self.client.rpush(self.key, *serialized)
        if self.ttl_seconds > 0:
            await self.client.expire(self.key, self.ttl_seconds)

    async def add_user_message(self, content: str) -> None:
        await self.add_messages([HumanMessage(content=content)])

    async def add_ai_message(self, content: str) -> None:
        await self.add_messages([AIMessage(content=content)])

    async def clear(self) -> None:
        await self.client.delete(self.key)


def get_redis_history(session_id: str) -> AsyncRedisChatMessageHistory:
    return AsyncRedisChatMessageHistory(session_id=session_id)


ROLE_MAP = {
    HumanMessage: "user",
    AIMessage: "assistant",
    SystemMessage: "system",
}


def to_api_messages(messages: list[BaseMessage]) -> list[dict]:
    output = []
    for msg in messages:
        role = ROLE_MAP.get(type(msg), "assistant")
        output.append({"role": role, "content": msg.content})
    return output
