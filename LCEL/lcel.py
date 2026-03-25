import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

load_dotenv()
qwen_api_key = os.getenv("DASHSCOPE_API_KEY")

# 1. 初始化核心组件
llm = ChatTongyi(
    model="qwen-plus",
    max_retries=2,
    api_key=qwen_api_key,
)
parser = StrOutputParser()

# ======================
# 一、基础链
# ======================
base_chain = (
        ChatPromptTemplate.from_template("把这句话翻译成英文：{text}")
        | llm
        | parser
)

# ======================
# 二、串行链（已修复！）
# ======================
summary_chain = (
        ChatPromptTemplate.from_template("请给下面文本生成摘要：{text}")
        | llm
        | parser
)

# 修复：变量名改成 input，直接接收上一步输出
keyword_chain = (
        ChatPromptTemplate.from_template("从内容中提取3个关键词：{input}")
        | llm
        | parser
)

# 串行组合（修复完成）
sequential_chain = summary_chain | keyword_chain

# ======================
# 三、并行链
# ======================
parallel_chain = {
    "original": RunnablePassthrough(),  # 并行链里保留「原始输入」
    "summary": summary_chain,
    "keywords": lambda x: keyword_chain.invoke({"input": summary_chain.invoke(x)})
}


# ======================
# 四、自定义函数
# ======================
def format_output(text):
    return f"【最终报告】\n{text.upper()}"


# ======================
# 五、终极混合链
# ======================
report_chain = (
        ChatPromptTemplate.from_template("""
    根据以下信息生成分析报告：
    原文：{original}
    摘要：{summary}
    关键词：{keywords}
    """)
        | llm
        | parser
)


def show(_text):
    print(f"****\n{_text}\n****")
    return _text


final_chain = parallel_chain | RunnableLambda(show) | report_chain | RunnableLambda(format_output)

# ----------------------
# 测试运行
# ----------------------
if __name__ == "__main__":
    test_text = """
    LangChain是一个大模型应用开发框架，
    支持快速构建LLM应用、RAG检索、智能体等功能，
    广泛用于问答、内容创作、自动化流程。
    """

    print("=" * 50)
    print("基础翻译：")
    print(base_chain.invoke({"text": "你好LCEL"}))

    print("\n" + "=" * 50)
    print("串行摘要→关键词：")
    print(sequential_chain.invoke({"text": test_text}))

    print("\n" + "=" * 50)
    print("终极流程：")
    print(final_chain.invoke({"text": test_text}))
