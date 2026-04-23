from typing import Any, Literal

from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader
from langchain_core.documents import Document


def load_pdf(
        file_path: str = "../assets/sample.pdf",  # PDF 文件路径（支持相对/绝对路径）
        mode: Literal["page", "single"] = "page",  # PyPDF 模式：按页/整文档
        password: str | bytes | None = None,  # 加密 PDF 密码
        extract_images: bool = False,  # 是否提取图片并交给 images_parser
        extraction_mode: Literal["plain", "layout"] = "plain",  # 文本提取模式
        pages_delimiter: str = "\n\f",  # single 模式下分页分隔符
        use_pypdf: bool = True,  # True: 优先 PyPDFLoader
        fallback_to_unstructured: bool = True,  # PyPDF 失败时是否降级 UnstructuredPDFLoader
        unstructured_mode: str = "elements",  # Unstructured 模式：single/elements
        **unstructured_kwargs: Any,  # 透传给 UnstructuredPDFLoader 的高级参数
) -> list[Document]:
    """PDF -> List[Document]

    常见骚操作示例：
    1) 每页切分：mode="page"
    2) 保留版面：extraction_mode="layout"
    3) 失败回退：fallback_to_unstructured=True
    """
    if use_pypdf:
        try:
            loader = PyPDFLoader(
                file_path=file_path,  # 文件路径
                password=password,  # 密码
                extract_images=extract_images,  # 图片提取
                mode=mode,  # page/single
                pages_delimiter=pages_delimiter,  # 分页分隔符
                extraction_mode=extraction_mode,  # plain/layout
            )
            return loader.load()
        except Exception as e:
            print(f"PyPDFLoader failed: {e}")
            if not fallback_to_unstructured:
                raise

    loader = UnstructuredPDFLoader(
        file_path=file_path,  # 文件路径
        mode=unstructured_mode,  # single/elements
        **unstructured_kwargs,  # 其他 unstructured 参数
    )
    return loader.load()


if __name__ == "__main__":
    print(load_pdf())
