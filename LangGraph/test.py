import os
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.runnables.graph import MermaidDrawMethod  # 画图用
from langchain_community.chat_models import ChatTongyi


# 状态：记录消息、当前代理、任务结果
class MultiAgentState(TypedDict):
    messages: Annotated[list, add_messages]
    current_agent: str
    task_result: dict


# LLM
llm = ChatTongyi(
    model=os.getenv("QWEN_MODEL", "qwen-plus"),
    api_key=os.getenv("DASHSCOPE_API_KEY","sk-e47084aadc1249cc8fdd5bb8db0678fc"),
)


# 代理1：研究员
def researcher(state: MultiAgentState):
    prompt = "作为研究员，分析问题并搜集关键信息：" + state["messages"][-1].content
    response = llm.invoke(prompt)
    return {"task_result": {"research": response.content}, "current_agent": "summarizer"}


# 代理2：总结员
def summarizer(state: MultiAgentState):
    prompt = "作为总结员，精简以下信息：" + state["task_result"]["research"]
    response = llm.invoke(prompt)
    return {"messages": [response], "current_agent": "end"}


# 路由
def next_agent(state: MultiAgentState):
    print(state,"??")
    return state["current_agent"]


# ====================== 构建图 ======================
builder = StateGraph(MultiAgentState)
builder.add_node("researcher", researcher)
builder.add_node("summarizer", summarizer)

builder.set_entry_point("researcher")

# 路由规则
builder.add_conditional_edges(
    "researcher",
    next_agent,
    {"summarizer": "summarizer"}  # 明确映射，防止连错
)
builder.add_conditional_edges(
    "summarizer",
    next_agent,
    {"end": END}
)

# 编译
graph = builder.compile()

# ====================== ✅ 自动画出流程图 ======================
# graph.get_graph().draw_mermaid_png(
#     draw_method=MermaidDrawMethod.API,
#     output_file_path="multi_agent_graph.png"
# )

# ====================== 运行 ======================
result = graph.invoke({
    "messages": [{"role": "user", "content": "AI未来发展趋势"}],
    "current_agent": "researcher"
})

print("最终总结:", result["messages"][-1].content)
