from langchain_community.chat_models import ChatTongyi

from app.conf.app_config import app_config

llm = ChatTongyi(
    model=app_config.llm.model_name,
    max_retries=2,
    api_key=app_config.llm.api_key,
    temperature=0
)

if __name__ == '__main__':
    for chunk in llm.stream("What is the meaning of life?"):
        print(chunk.text)
