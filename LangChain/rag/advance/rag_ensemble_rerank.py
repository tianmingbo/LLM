"""
混合召回 + Cross-Encoder 重排

流程：
1) 先用 EnsembleRetriever 做混合召回（BM25 + 向量检索）
2) 再用 HuggingFaceCrossEncoder 对候选文档重排
3) 取重排后 TopN 作为上下文，交给 LLM 回答
"""

import os

from pymilvus import MilvusClient
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_community.chat_models import ChatTongyi
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_huggingface import HuggingFaceEmbeddings


# ========== 向量检索器封装 ==========
class MilvusRetriever(BaseRetriever):
    """将 MilvusClient.search 封装成 LangChain Retriever。"""

    client: MilvusClient
    embed_model: HuggingFaceEmbeddings
    collection_name: str = "demo_collection"
    anns_field: str = "vector"
    metric_type: str = "L2"
    limit: int = 8  # 召回阶段建议多取一点，给 rerank 留空间

    def _get_relevant_documents(self, query: str) -> list[Document]:
        query_embedding = self.embed_model.embed_query(query)
        hits = self.client.search(
            collection_name=self.collection_name,
            data=[query_embedding],
            anns_field=self.anns_field,
            search_params={"metric_type": self.metric_type},
            output_fields=["text", "metadata"],
            limit=self.limit,
        )

        docs = []
        for item in hits[0]:
            entity = item.get("entity", {})
            text = entity.get("text", "")
            if not text:
                continue

            metadata = entity.get("metadata", {})
            metadata["vector_distance"] = item.get("distance")
            docs.append(Document(page_content=text, metadata=metadata))
        return docs


def _build_bm25_retriever(client: MilvusClient, limit: int = 300) -> BM25Retriever:
    """从向量库抽样文本，构建 BM25 检索器语料。"""
    rows = client.query(
        collection_name="demo_collection",
        output_fields=["text", "metadata"],
        limit=limit,
    )
    docs = [
        Document(page_content=row.get("text", ""), metadata=row.get("metadata", {}))
        for row in rows
        if row.get("text")
    ]

    retriever = BM25Retriever.from_documents(docs)
    retriever.k = 8
    return retriever


def get_optimal_weights(data_type: str) -> list[float]:
    """根据数据特点选择融合权重。"""
    weights_map = {
        "technical_docs": [0.4, 0.6],
        "code_base": [0.6, 0.4],
        "mixed": [0.5, 0.5],
        "conversation": [0.3, 0.7],
    }
    return weights_map.get(data_type, [0.5, 0.5])


def build_ensemble_retriever(embed_model: HuggingFaceEmbeddings, client: MilvusClient) -> EnsembleRetriever:
    """构建融合检索器：BM25 + Milvus 向量检索。"""
    bm25_retriever = _build_bm25_retriever(client=client)
    milvus_retriever = MilvusRetriever(
        client=client,
        embed_model=embed_model,
        collection_name="demo_collection",
        anns_field="vector",
        metric_type="L2",
        limit=8,
    )
    return EnsembleRetriever(
        retrievers=[bm25_retriever, milvus_retriever],
        weights=get_optimal_weights("technical_docs"),
    )


def rerank_docs(query: str, docs: list[Document], reranker: HuggingFaceCrossEncoder, top_n: int = 3) -> list[Document]:
    """用 Cross-Encoder 对融合召回结果重排。"""
    if not docs:
        return []

    pairs = [(query, d.page_content) for d in docs]
    scores = reranker.score(pairs)

    scored_docs: list[tuple[float, Document]] = []
    for doc, score in zip(docs, scores):
        doc.metadata = {**doc.metadata, "rerank_score": round(float(score), 6)}
        scored_docs.append((float(score), doc))

    scored_docs.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in scored_docs[:top_n]]


def retrieval(query: str, retriever: EnsembleRetriever, reranker: HuggingFaceCrossEncoder) -> str:
    """完整检索流程：混合召回 -> Cross-Encoder 重排 -> 拼接上下文。"""
    recalled_docs = retriever.invoke(query)
    reranked_docs = rerank_docs(query=query, docs=recalled_docs, reranker=reranker, top_n=3)

    print("\n===== Rerank Top3 =====")
    for idx, doc in enumerate(reranked_docs, 1):
        print(
            f"[{idx}] rerank_score={doc.metadata.get('rerank_score')} "
            f"vector_distance={doc.metadata.get('vector_distance')}"
        )

    return "\n\n".join(f"[{idx}] {doc.page_content}" for idx, doc in enumerate(reranked_docs, 1))


# ========== 运行入口 ==========
client = MilvusClient(uri="./milvus_demo.db")

embed_model = HuggingFaceEmbeddings(
    model_name=os.path.expanduser("~/models/bge-base-zh-v1.5")
)

# 使用 LangChain 封装的 CrossEncoder（底层依赖 sentence-transformers）
reranker = HuggingFaceCrossEncoder(
    model_name=os.path.expanduser("~/models/bge-reranker-base")
)

ensemble_retriever = build_ensemble_retriever(embed_model=embed_model, client=client)

llm = ChatTongyi(
    model=os.getenv("QWEN_MODEL", "qwen-plus"),
    api_key=os.getenv("DASHSCOPE_API_KEY"),
)

template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "# 任务\n\n根据上下文参考，回答用户的问题。\n\n# 上下文参考\n\n{context}",
        ),
        ("human", "{query}"),
    ]
)

rag_chain = (
        {
            "query": RunnablePassthrough(),
            "context": lambda x: retrieval(query=x, retriever=ensemble_retriever, reranker=reranker),
        }
        | RunnableLambda(lambda x: print(x) or x)
        | template
        | llm
        | StrOutputParser()
)

if __name__ == "__main__":
    res_chunks = rag_chain.stream(input="不动产被占有了怎么办?")
    for chunk in res_chunks:
        print(chunk, end="", flush=True)
    print()
