"""
流式输出
"""
import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatTongyi
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv(override=True)
qwen_api_key = os.getenv("DASHSCOPE_API_KEY")

# 初始化 deepseek
model = ChatTongyi(
    model="qwen-plus",
    max_retries=2,
    api_key=qwen_api_key,
)
messages = [
    SystemMessage(content="你是小黑子，会唱跳rap"),
    HumanMessage(content="你是谁？"),
]
response = model.stream(messages)
# 流式打印结果
for chunk in response:
    print(chunk.content, end="", flush=True)
print("\n")
print(type(response))  # 生成器
