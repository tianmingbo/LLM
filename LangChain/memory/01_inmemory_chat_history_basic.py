"""示例 1（create_agent）：直接使用 InMemoryChatMessageHistory 存储多轮对话"""

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
    agent = create_agent(model=model, tools=[], system_prompt="你是一个简洁的助手。", checkpointer=checkpointer)

    res_1 = agent.invoke(
        {"messages": [{"role": "user", "content": "我叫张三"}]},
        config={"configurable": {"thread_id": "basic_demo"}})
    ai_msg_1 = res_1["messages"][-1]
    print("第一轮:", ai_msg_1.content, "\n")

    res_2 = agent.invoke(
        {"messages": [{"role": "user", "content": "who am i？"}]},
        config={"configurable": {"thread_id": "basic_demo"}})
    ai_msg_2 = res_2["messages"][-1]
    print("第二轮:", ai_msg_2.content, "\n")


if __name__ == "__main__":
    main()
