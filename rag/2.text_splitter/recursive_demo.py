from langchain_text_splitters import RecursiveCharacterTextSplitter

sample_text = """
# 部署指南

## 1. 环境准备
请先安装 Python 3.11，并创建虚拟环境。随后安装依赖并校验版本。

## 2. 配置模型
在 .env 中填写 API_KEY，并配置基础 URL。建议先用小模型完成联调，再切换到生产模型。

## 3. 启动服务
执行 uvicorn app:app --reload 启动服务。若端口冲突，请改为 8001 或 9000。
""".strip()

splitter = RecursiveCharacterTextSplitter(
    separators=["\n## ", "\n\n", "\n", "。", "，", " ", ""],
    chunk_size=80,
    chunk_overlap=20,
)

chunks = splitter.split_text(sample_text)

for idx, chunk in enumerate(chunks, start=1):
    print(f"[{idx}] {chunk}\n")
