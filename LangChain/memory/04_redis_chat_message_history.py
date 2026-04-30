"""示例 5（create_agent）：Redis checkpointer（持久化）"""

import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langgraph.checkpoint.redis import RedisSaver
from langchain_community.chat_models import ChatTongyi

load_dotenv(override=True)
qwen_api_key = os.getenv("DASHSCOPE_API_KEY")


def main() -> None:
    model = ChatTongyi(model="qwen-plus", max_retries=2, api_key=qwen_api_key)

    redis_url = os.getenv("REDIS_URL", "redis://:tian666@localhost:6389/0")

    # 使用 Redis 持久化 checkpoint，跨进程/多实例可共享会话记忆
    with RedisSaver.from_conn_string(redis_url) as checkpointer:
        checkpointer.setup()
        agent = create_agent(model=model, tools=[], system_prompt="你是一个助手。", checkpointer=checkpointer)

        config = {"configurable": {"thread_id": "redis_user_1"}}

        agent.invoke({"messages": [{"role": "user", "content": "我在北京工作"}]}, config=config)
        result = agent.invoke({"messages": [{"role": "user", "content": "我在哪里工作？"}]}, config=config)
        print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
