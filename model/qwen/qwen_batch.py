"""
批量请求 + 每条答案流式输出
"""
import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatTongyi
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv(override=True)
qwen_api_key = os.getenv("DASHSCOPE_API_KEY")

# 初始化 qwen
model = ChatTongyi(
    model="qwen-plus",
    max_retries=2,
    api_key=qwen_api_key,
)

batch_messages = [
    [
        SystemMessage(content="你是小黑子，会唱跳rap"),
        HumanMessage(content="你是谁？"),
    ],
    [
        SystemMessage(content="你是中黑子，会唱跳rap"),
        HumanMessage(content="你是谁？"),
    ],
    [
        SystemMessage(content="你是大黑子，会唱跳rap"),
        HumanMessage(content="你是谁？"),
    ],
]

# batch 接口会等所有结果完成后一次性返回；
# 这里改成“批量输入逐条 stream”，保证每个答案都实时输出。
for idx, one_round_messages in enumerate(batch_messages, start=1):
    print(f"\n==== 第{idx}条开始 ====")
    print(f"q: {one_round_messages[-1].content}")
    print("r: ", end="", flush=True)
    for chunk in model.stream(one_round_messages):
        if chunk.content:
            print(chunk.content, end="", flush=True)
    print("\n==== 第{idx}条结束 ====\n")
