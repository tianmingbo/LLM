"""改执行流程/状态"""

import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_community.chat_models import ChatTongyi
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv(override=True)
qwen_api_key = os.getenv("DASHSCOPE_API_KEY")


class MyMiddleware(AgentMiddleware):
    def before_model(self, state, runtime):
        """模型调用前执行"""
        print("准备调用模型")
        return None  # 返回 None 表示继续正常流程

    def after_model(self, state, runtime):
        """模型响应后执行"""
        print("模型已响应")
        return None  # 返回 None 表示不修改状态


def main() -> None:
    model = ChatTongyi(model="qwen-plus", max_retries=2, api_key=qwen_api_key)
    checkpointer = InMemorySaver()

    agent = create_agent(
        model=model,
        tools=[],
        system_prompt="你是一个简洁助手。",
        middleware=[MyMiddleware()],
        checkpointer=checkpointer,
    )

    config = {"configurable": {"thread_id": "mw_log_demo"}}
    agent.invoke({"messages": [{"role": "user", "content": "你好，我叫小明"}]}, config=config)
    result = agent.invoke({"messages": [{"role": "user", "content": "我叫什么？"}]}, config=config)
    print("最终回答:", result["messages"][-1].content)


if __name__ == "__main__":
    main()
