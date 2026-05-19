import asyncio
import os
from http import HTTPStatus
from typing import Optional

import dashscope

from app.conf.app_config import EmbeddingConfig, app_config


class QwenEmbeddingClient:
    def __init__(
        self,
        model_name: str,
        dimension: int = 1024,
        api_key: str | None = None,
        base_http_api_url: str | None = None,
    ):
        self.model_name = model_name
        self.dimension = dimension
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("DashScope API key is required: set embedding.api_key or DASHSCOPE_API_KEY")
        if base_http_api_url:
            dashscope.base_http_api_url = base_http_api_url
        dashscope.api_key = self.api_key

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        resp = dashscope.TextEmbedding.call(
            model=self.model_name,
            input=texts,
            text_type="document",
            dimension=self.dimension,
        )
        if resp.status_code != HTTPStatus.OK:
            raise RuntimeError(f"DashScope embedding failed: {resp.code} {resp.message}")
        return [item["embedding"] for item in resp.output["embeddings"]]

    def embed_query(self, text: str) -> list[float]:
        resp = dashscope.TextEmbedding.call(
            model=self.model_name,
            input=text,
            text_type="query",
            dimension=self.dimension,
        )
        if resp.status_code != HTTPStatus.OK:
            raise RuntimeError(f"DashScope embedding failed: {resp.code} {resp.message}")
        return resp.output["embeddings"][0]["embedding"]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self.embed_documents, texts)

    async def aembed_query(self, text: str) -> list[float]:
        return await asyncio.to_thread(self.embed_query, text)


class EmbeddingClientManager:
    def __init__(self, config: EmbeddingConfig):
        self.client: Optional[QwenEmbeddingClient] = None
        self.config = config

    def init(self):
        self.client = QwenEmbeddingClient(
            model_name=self.config.model,
            dimension=self.config.dimension,
            api_key=self.config.api_key,
            base_http_api_url=self.config.base_http_api_url,
        )


embedding_client_manager = EmbeddingClientManager(app_config.embedding)
