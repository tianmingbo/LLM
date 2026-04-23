from typing import Any

from langchain_community.document_loaders import TextLoader, UnstructuredMarkdownLoader
from langchain_core.documents import Document


def load_markdown(
        file_path: str = "../assets/sample.md",  # Markdown 文件路径（支持相对/绝对路径）
        mode: str = "elements",  # Unstructured 模式：single/elements
        encoding: str = "utf-8",  # 文件编码（TextLoader 及兜底场景使用）
        autodetect_encoding: bool = False,  # 是否自动探测编码（TextLoader）
        use_unstructured: bool = True,  # True: 优先 UnstructuredMarkdownLoader；False: 直接 TextLoader
        fallback_to_text: bool = True,  # Unstructured 失败时是否降级到 TextLoader
        **unstructured_kwargs: Any,  # 透传给 UnstructuredMarkdownLoader 的高级参数
) -> list[Document]:
    """Markdown -> List[Document]

    常见骚操作示例：
    1) 按块拆分：mode="elements"
    2) 透传参数：strategy="fast"（依赖 unstructured 版本支持）
    3) 关闭 Unstructured：use_unstructured=False（纯文本直读）
    """
    if use_unstructured:
        try:
            loader = UnstructuredMarkdownLoader(
                file_path=file_path,  # 文件路径
                mode=mode,  # single/elements
                **unstructured_kwargs,  # 其他 unstructured 参数
            )
            return loader.load()
        except Exception as e:
            print(f"UnstructuredMarkdownLoader failed: {e}")
            if not fallback_to_text:
                raise

    loader = TextLoader(
        file_path=file_path,  # 文件路径
        encoding=encoding,  # 文件编码
        autodetect_encoding=autodetect_encoding,  # 自动编码探测
    )
    return loader.load()


if __name__ == "__main__":
    res = load_markdown(mode="single")
    print(len(res))
    print(res)
