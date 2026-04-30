"""
监听事件/打点
"""
import os
from uuid import UUID
import dotenv
from typing import Dict, Any, Optional, List
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_community.chat_models import ChatTongyi


class CustomCallbackHandler(BaseCallbackHandler):
    """自定义回调处理类"""

    def on_chat_model_start(self, serialized: Dict[str, Any], messages: List[List[BaseMessage]], *, run_id: UUID,
                            parent_run_id: Optional[UUID] = None, tags: Optional[List[str]] = None,
                            metadata: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
        print("======聊天模型开始执行======")

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, parent_run_id: Optional[UUID] = None,
                   **kwargs: Any) -> Any:
        print("======聊天模型结束执行======")

    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], *, run_id: UUID,
                       parent_run_id: Optional[UUID] = None, tags: Optional[List[str]] = None,
                       metadata: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
        print(f"开始执行当前组件{kwargs['name']}，run_id: {run_id}, 入参：{inputs}")

    def on_chain_end(self, outputs: Dict[str, Any], *, run_id: UUID, parent_run_id: Optional[UUID] = None,
                     **kwargs: Any) -> Any:
        print(f"结束执行当前组件，run_id: {run_id}, 执行结果：{outputs}, {kwargs}")


# 读取env配置
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

# 创建 Chain
chain = prompt | model
# 设置回调处理类
config = RunnableConfig(callbacks=[CustomCallbackHandler()])
# 打印结果
chain.invoke({"question": "什么是LangChain?"}, config)
