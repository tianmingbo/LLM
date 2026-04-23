"""
固定长度切分
"""

from langchain_text_splitters import CharacterTextSplitter

sample_text = """
RAG 文本切分的目标不是平均长度，而是保证语义完整和可检索性。

当 chunk 过大时，召回阶段会混入无关信息；当 chunk 过小时，答案需要的上下文又可能被切断。

一个常见起点是 chunk_size=500，chunk_overlap=50，然后结合实际问答样本做迭代。
""".strip()

splitter = CharacterTextSplitter(
    separator="",  # 纯固定长度切分（不按段落边界）
    chunk_size=40,
    chunk_overlap=10,
)

chunks = splitter.split_text(sample_text)

for idx, chunk in enumerate(chunks, start=1):
    print(f"[{idx}] {chunk}\n")
