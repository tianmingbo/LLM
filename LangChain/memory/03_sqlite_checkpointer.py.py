"""示例 4（create_agent）：Sqlite checkpointer（本地持久化）"""

import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_community.chat_models import ChatTongyi

load_dotenv(override=True)
qwen_api_key = os.getenv("DASHSCOPE_API_KEY")


def main() -> None:
    model = ChatTongyi(model="qwen-plus", max_retries=2, api_key=qwen_api_key)

    # 使用本地 sqlite 文件持久化 checkpoint，重启进程后仍可读取同一 thread_id 的记忆
    with SqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
        agent = create_agent(model=model, tools=[], system_prompt="你是一个助手。", checkpointer=checkpointer)

        config = {"configurable": {"thread_id": "sqlite_demo"}}

        agent.invoke({"messages": [{"role": "user", "content": "我喜欢篮球"}]}, config=config)
        result = agent.invoke({"messages": [{"role": "user", "content": "我喜欢什么运动？"}]}, config=config)
        print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
