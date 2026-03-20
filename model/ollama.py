from langchain_community.chat_models import ChatOllama
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# 设置本地模型，不使用深度思考
model = ChatOllama(base_url="http://localhost:11434", model="qwen:7b", temperature=0.7)

messages = [
    SystemMessage(content="你是谁？"),
    HumanMessage(content="什么是LangChain?"),
]

response = model.invoke(messages)
print(response.content)
