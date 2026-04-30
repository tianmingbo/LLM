"""生成摘要"""

import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langchain_community.chat_models import ChatTongyi

load_dotenv(override=True)
qwen_api_key = os.getenv("DASHSCOPE_API_KEY")


def main() -> None:
    model = ChatTongyi(model="qwen-plus", max_retries=2, api_key=qwen_api_key)
    checkpointer = InMemorySaver()
    agent = create_agent(
        model=model,
        system_prompt="你是一个简洁助手。",
        checkpointer=checkpointer,
        middleware=[
            SummarizationMiddleware(
                model=model,
                trigger=('tokens', 500)  # 超过 500 tokens 就摘要
            )
        ]
    )

    config = {"configurable": {"thread_id": "trim_demo"}}

    agent.invoke({"messages": [{"role": "user", "content": "我叫小明"}]}, config=config)
    agent.invoke({"messages": [{"role": "user", "content": "生成1000字自我介绍"}]}, config=config)
    agent.invoke({"messages": [{"role": "user", "content": "我住在上海"}]}, config=config)

    for i in checkpointer.list(config=config):
        print(i,'\n')


if __name__ == "__main__":
    main()
