"""
结构化输出
"""
import os
from typing import List
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_community.chat_models import ChatTongyi

load_dotenv(override=True)
qwen_api_key = os.getenv("DASHSCOPE_API_KEY")


class Invoice(BaseModel):
    invoice_number: str = Field(description="发票号")
    date: str = Field(description="日期")
    total_amount: float = Field(description="金额")
    items: List[str] = Field(description="商品")


model = ChatTongyi(model="qwen-plus", max_retries=2, api_key=qwen_api_key)

structured_llm = model.with_structured_output(Invoice)

invoice_text = """
发票号: INV-2024-001
日期: 2024-01-15
总金额: 1299.00
商品: MacBook Pro, AppleCare+
"""

invoice = structured_llm.invoke(f"提取发票信息：{invoice_text}")
# invoice.invoice_number = "INV-2024-001"
# invoice.total_amount = 1299.00
print(invoice)
