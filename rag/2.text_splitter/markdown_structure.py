"""
结构化（Markdown 标题）切分
"""
from langchain_text_splitters import MarkdownHeaderTextSplitter

sample_markdown = """
# 产品手册

## 快速开始
安装依赖后运行 `python main.py`。

## 检索参数
### chunk_size
推荐从 400 到 800 token 开始。

### chunk_overlap
通常设置为 chunk_size 的 10% 到 20%。

## 常见问题
如果召回偏低，先检查切分是否过碎。
""".strip()

headers_to_split_on = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
]

splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
docs = splitter.split_text(sample_markdown)

for idx, doc in enumerate(docs, start=1):
    print(f"[{idx}] metadata={doc.metadata}")
    print(doc.page_content)
    print()
