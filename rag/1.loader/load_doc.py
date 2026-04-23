from typing import Any

from langchain_community.document_loaders import Docx2txtLoader, UnstructuredWordDocumentLoader
from langchain_core.documents import Document


def load_doc(
        file_path: str = "../assets/sample.docx",  # Word 文件路径（.doc/.docx）
        mode: str = "elements",  # Unstructured 模式：single/elements
        use_unstructured: bool = True,  # True: 优先 UnstructuredWordDocumentLoader
        fallback_to_docx2txt: bool = True,  # Unstructured 失败时，.docx 是否降级到 Docx2txtLoader
        **unstructured_kwargs: Any,  # 透传给 UnstructuredWordDocumentLoader 的高级参数
) -> list[Document]:
    """DOC/DOCX -> List[Document]

    常见骚操作示例：
    1) 结构化切分：mode="elements"
    2) 单文档模式：mode="single"
    3) 依赖不完整时：自动回退 Docx2txt（仅 .docx）
    """
    if use_unstructured:
        try:
            loader = UnstructuredWordDocumentLoader(
                file_path=file_path,  # 文件路径
                mode=mode,  # single/elements
                **unstructured_kwargs,  # 其他 unstructured 参数
            )
            return loader.load()
        except Exception as e:
            print(f"UnstructuredWordDocumentLoader failed: {e}")
            if not fallback_to_docx2txt:
                raise

    if file_path.lower().endswith(".docx"):
        loader = Docx2txtLoader(file_path=file_path)  # .docx 回退路径
        return loader.load()

    raise RuntimeError(
        "DOC 文件当前仅支持 Unstructured 路径，请安装 unstructured 相关依赖或改用 .docx。"
    )


if __name__ == "__main__":
    print(load_doc())
