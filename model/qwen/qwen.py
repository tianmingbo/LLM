"""
基础对话
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
response = model.invoke(messages)
print(response.content)
