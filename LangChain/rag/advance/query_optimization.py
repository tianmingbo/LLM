"""
查询优化（Query Optimization）

目标：
- 在进入检索前，先对用户原始问题做轻量优化，提升召回质量。
- 这里演示 3 个常见策略：
  1) Query Rewrite（规范化改写）
  2) Multi-Query Expansion（多路查询扩展）
  3) HyDE 风格假设答案（可选）

说明：
- 为了可运行和易理解，示例不依赖额外复杂组件。
- 你可以直接把这些函数接到已有的 rag_ensemble / rag_ensemble_rerank 流程前面。
"""

import os
import re
from typing import List

from langchain_community.chat_models import ChatTongyi
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda


# ========== 1) 规则级 Query Rewrite ==========
def normalize_query(query: str) -> str:
    """对原始 query 做低风险规范化。

    规则：
    - 去掉多余空白
    - 全角标点转半角（常见几种）
    - 常见口语词替换为检索友好词
    """
    q = query.strip()
    q = q.replace("，", ",").replace("。", ".").replace("？", "?")
    q = re.sub(r"\s+", " ", q)

    # 这里仅做示例，生产环境可维护成同义词词典
    replacements = {
        "怎么办": "处理流程",
        "咋办": "处理流程",
        "啥": "什么",
    }
    for src, dst in replacements.items():
        q = q.replace(src, dst)
    return q


# ========== 2) LLM 生成多路查询（Multi-Query） ==========
def generate_multi_queries(query: str, llm: ChatTongyi, max_queries: int = 3) -> List[str]:
    """让 LLM 生成多个等价检索问法，用于提升召回覆盖率。"""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是检索优化助手。请把用户问题改写成多个适合检索的问法，"
                "每行一条，不要编号，不要解释。",
            ),
            (
                "human",
                "原问题：{query}\n"
                "请给出 {max_queries} 条检索问法，覆盖："
                "关键词表达、法条表达、场景表达。",
            ),
        ]
    )

    chain = prompt | llm | StrOutputParser()
    raw = chain.invoke({"query": query, "max_queries": max_queries})

    lines = [line.strip(" -\t") for line in raw.splitlines() if line.strip()]
    # 去重并截断到 max_queries
    uniq = []
    seen = set()
    for line in lines:
        if line not in seen:
            seen.add(line)
            uniq.append(line)
        if len(uniq) >= max_queries:
            break

    return uniq or [query]


# ========== 3) HyDE：生成“假设答案”用于语义召回 ==========
def generate_hypothetical_answer(query: str, llm: ChatTongyi) -> str:
    """生成一段简短“假设答案”，可拿去做 embedding 检索（HyDE 思路）。"""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是检索预处理助手。基于问题写一段简短、客观、结构化的假设答案，"
                "用于向量检索，不要编造细节。",
            ),
            ("human", "问题：{query}"),
        ]
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"query": query}).strip()


# ========== 演示：把优化结果拼成“检索计划” ==========
def build_query_plan(raw_query: str, llm: ChatTongyi) -> dict:
    """返回一个可直接接入检索层的查询计划。"""
    rewritten = normalize_query(raw_query)
    multi_queries = generate_multi_queries(rewritten, llm=llm, max_queries=3)
    hyde_text = generate_hypothetical_answer(rewritten, llm=llm)

    return {
        "raw_query": raw_query,
        "rewritten_query": rewritten,
        "multi_queries": multi_queries,
        "hyde_text": hyde_text,
    }


def fake_retrieve(query: str) -> list[Document]:
    """仅用于演示：实际项目中替换为 BM25/向量/混合召回。"""
    return [
        Document(page_content=f"[模拟召回] 命中内容 A（query={query}）"),
        Document(page_content=f"[模拟召回] 命中内容 B（query={query}）"),
    ]


def retrieve_with_query_plan(query_plan: dict) -> list[Document]:
    """把多个优化 query 的召回结果合并去重。"""
    all_docs: list[Document] = []

    # 主查询（rewrite 后）
    all_docs.extend(fake_retrieve(query_plan["rewritten_query"]))

    # 多路扩展查询
    for q in query_plan["multi_queries"]:
        all_docs.extend(fake_retrieve(q))

    # HyDE 文本也作为一次检索输入（常用于向量检索）
    all_docs.extend(fake_retrieve(query_plan["hyde_text"]))

    # 按 page_content 去重
    dedup = {}
    for d in all_docs:
        dedup[d.page_content] = d
    return list(dedup.values())


def main() -> None:
    llm = ChatTongyi(
        model=os.getenv("QWEN_MODEL", "qwen-plus"),
        api_key=os.getenv("DASHSCOPE_API_KEY"),
    )

    raw_query = "不动产被占有了怎么办？"
    plan = build_query_plan(raw_query=raw_query, llm=llm)

    print("\n===== Query Plan =====")
    print("raw_query:", plan["raw_query"])
    print("rewritten_query:", plan["rewritten_query"])
    print("multi_queries:")
    for i, q in enumerate(plan["multi_queries"], 1):
        print(f"  {i}. {q}")
    print("hyde_text:", plan["hyde_text"])

    docs = retrieve_with_query_plan(plan)
    print("\n===== Merged Retrieval Results =====")
    for i, d in enumerate(docs, 1):
        print(f"[{i}] {d.page_content}")


if __name__ == "__main__":
    main()
