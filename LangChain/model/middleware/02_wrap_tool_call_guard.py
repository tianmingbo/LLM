"""示例 2：wrap_tool_call 拦截工具调用参数"""

import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_community.chat_models import ChatTongyi
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv(override=True)
qwen_api_key = os.getenv("DASHSCOPE_API_KEY")


@tool
def get_salary(name: str) -> str:
    """查询员工薪资（演示工具）。"""
    fake_db = {"张三": "30000", "李四": "28000"}
    value = fake_db.get(name, "未知")
    return f"{name} 的薪资是 {value} 元"


@wrap_tool_call
def deny_sensitive_name(request, handler):
    args = request.tool_call.get("args", {})
    if args.get("name") == "张三":
        return ToolMessage(
            content="该姓名属于敏感信息，已拒绝查询。",
            tool_call_id=request.tool_call["id"],
        )
    return handler(request)


def main() -> None:
    model = ChatTongyi(model="qwen-plus", max_retries=2, api_key=qwen_api_key)
    checkpointer = InMemorySaver()

    agent = create_agent(
        model=model,
        tools=[get_salary],
        system_prompt="你是一个助手，需要时可以调用工具。",
        middleware=[deny_sensitive_name],
        checkpointer=checkpointer,
    )

    config = {"configurable": {"thread_id": "mw_tool_guard_demo"}}
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "帮我查一下张三的薪资"}]},
        config=config,
    )
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
