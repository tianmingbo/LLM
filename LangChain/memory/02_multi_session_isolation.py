"""示例 3（create_agent）：多会话隔离（不同 thread_id 的记忆互不影响）"""

import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_community.chat_models import ChatTongyi

load_dotenv(override=True)
qwen_api_key = os.getenv("DASHSCOPE_API_KEY")


def main() -> None:
    model = ChatTongyi(model="qwen-plus", max_retries=2, api_key=qwen_api_key)
    checkpointer = InMemorySaver()
    agent = create_agent(model=model, tools=[], system_prompt="你是一个助手。", checkpointer=checkpointer)

    config_a = {"configurable": {"thread_id": "user_A"}}
    config_b = {"configurable": {"thread_id": "user_B"}}

    agent.invoke({"messages": [{"role": "user", "content": "我叫小明"}]}, config=config_a)
    agent.invoke({"messages": [{"role": "user", "content": "我叫小红"}]}, config=config_b)

    ans_a = agent.invoke({"messages": [{"role": "user", "content": "我叫什么？"}]}, config=config_a)
    ans_b = agent.invoke({"messages": [{"role": "user", "content": "我叫什么？"}]}, config=config_b)

    print("A:", ans_a["messages"][-1].content)
    print("B:", ans_b["messages"][-1].content)


if __name__ == "__main__":
    main()
