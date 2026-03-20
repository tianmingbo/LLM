import os
from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek

load_dotenv(override=True)
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")

# 初始化 deepseek
model = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=deepseek_api_key,
)

# 打印结果
print(model.invoke("什么是LangChain?"))
