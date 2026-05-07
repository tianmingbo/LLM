# 读取env配置
import os

import dotenv
from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate

dotenv.load_dotenv()
# 构建 prompt 模板
template = """
    使用中文回答下面的问题：
    问题: {question}
    """
prompt = ChatPromptTemplate.from_template(template)

model = ChatTongyi(
    model=os.getenv("QWEN_MODEL", "qwen-plus"),
    api_key=os.getenv("DASHSCOPE_API_KEY"),
)

llm_with_retry = model.with_retry(
    retry_if_exception_type=(ConnectionError, TimeoutError),  # 重试的异常类型
    wait_exponential_jitter=True,  # 指数退避 + 随机抖动
    stop_after_attempt=3  # 最多重试 3 次
)

try:
    print("\n调用 LLM (如果失败会自动重试)...")
    response = llm_with_retry.invoke("你好")
    print(f"响应: {response.content[:50]}...")
    print("\n✓ 调用成功")
except Exception as e:
    print(f"\n✗ 重试 3 次后仍然失败: {e}")
