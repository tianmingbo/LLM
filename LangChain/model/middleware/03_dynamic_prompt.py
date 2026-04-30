"""示例 3：dynamic_prompt 根据会话状态动态设置系统提示词"""

import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt
from langchain_community.chat_models import ChatTongyi
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv(override=True)
qwen_api_key = os.getenv("DASHSCOPE_API_KEY")


@dynamic_prompt
def my_dynamic_prompt(request):
    msg_count = len(request.state["messages"])
    if msg_count >= 3:
        return "你在长对话中，请用一句话简洁回答。"
    return "你是一个耐心的中文助手。"


def main() -> None:
    model = ChatTongyi(model="qwen-plus", max_retries=2, api_key=qwen_api_key)
    checkpointer = InMemorySaver()

    agent = create_agent(
        model=model,
        tools=[],
        middleware=[my_dynamic_prompt],
        checkpointer=checkpointer,
    )

    config = {"configurable": {"thread_id": "mw_prompt_demo"}}
    agent.invoke({"messages": [{"role": "user", "content": "我叫小明"}]}, config=config)
    agent.invoke({"messages": [{"role": "user", "content": "我在上海"}]}, config=config)
    agent.invoke({"messages": [{"role": "user", "content": "1000字自我介绍"}]}, config=config)
    result = agent.invoke({"messages": [{"role": "user", "content": "总结一下我是谁"}]}, config=config)
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
