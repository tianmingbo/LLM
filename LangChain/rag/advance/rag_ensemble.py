"""
混合检索
BM25 检索（Keyword Search）
原理：基于词频(TF-IDF的改进版)，计算词的重要性

优势:
✅ 精确匹配专有名词
✅ 代码、版本号查询准确
✅ 速度快，无需嵌入


向量检索（Vector Search / Semantic Search）
原理：将文本转为向量，计算余弦相似度

优势：
✅ 理解语义和同义词
✅ 处理概念性查询
✅ 跨语言查询
"""

import os
from pymilvus import MilvusClient
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_community.chat_models import ChatTongyi
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_huggingface import HuggingFaceEmbeddings


# ========== 向量检索器封装 ==========
class MilvusRetriever(BaseRetriever):
    """将 MilvusClient.search 封装成 LangChain Retriever"""

    client: MilvusClient
    embed_model: HuggingFaceEmbeddings
    collection_name: str = "demo_collection"
    anns_field: str = "vector"
    metric_type: str = "L2"
    limit: int = 3

    def _get_relevant_documents(self, query: str) -> list[Document]:
        query_embedding = self.embed_model.embed_query(query)  # 查询嵌入
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
            metadata = entity.get("metadata", {})
            metadata["distance"] = item.get("distance")
            docs.append(Document(page_content=text, metadata=metadata))
        return docs


def _build_bm25_retriever(client: MilvusClient, limit: int = 200) -> BM25Retriever:
    """从向量库抽样一批文本，构建 BM25 检索器语料"""
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
    retriever.k = 3
    return retriever


# 根据数据类型选择权重
def get_optimal_weights(data_type):
    weights_map = {
        "technical_docs": [0.4, 0.6],  # 偏向语义
        "code_base": [0.6, 0.4],  # 偏向精确
        "mixed": [0.5, 0.5],  # 平衡
        "conversation": [0.3, 0.7],  # 强语义
    }
    return weights_map.get(data_type, [0.5, 0.5])


def build_ensemble_retriever(embed_model: HuggingFaceEmbeddings, client: MilvusClient) -> EnsembleRetriever:
    """构建融合检索器：BM25 + Milvus 向量检索"""
    bm25_retriever = _build_bm25_retriever(client=client)
    milvus_retriever = MilvusRetriever(
        client=client,
        embed_model=embed_model,
        collection_name="demo_collection",
        anns_field="vector",
        metric_type="L2",
        limit=3,
    )
    return EnsembleRetriever(
        retrievers=[bm25_retriever, milvus_retriever],
        weights=get_optimal_weights("technical_docs"),
    )


def retrieval(query: str, retriever: EnsembleRetriever) -> str:
    """检索并拼接上下文"""
    docs = retriever.invoke(query)
    context = []
    for idx, doc in enumerate(docs, 1):
        context.append(f"[{idx}] {doc.page_content}")
    return "\n\n".join(context)


# 实例化向量数据库客户端
client = MilvusClient(
    uri="./milvus_demo.db",  # 数据存储在本地当前目录下
)

# 加载嵌入模型
embed_model = HuggingFaceEmbeddings(
    model_name=os.path.expanduser("~/models/bge-base-zh-v1.5")
)

# 构建融合检索器
ensemble_retriever = build_ensemble_retriever(embed_model=embed_model, client=client)

# ========== 生成 ==========
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
            "context": lambda x: retrieval(query=x, retriever=ensemble_retriever),
        }
        | RunnableLambda(lambda x: print(x) or x)  # 打印中间结果
        | template
        | llm
        | StrOutputParser()
)

res_chunks = rag_chain.stream(input="不动产被占有了怎么办?")
for chunk in res_chunks:
    print(chunk, end="", flush=True)
