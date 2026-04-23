from langchain_community.document_loaders import BSHTMLLoader
from langchain_core.documents import Document


def load_html(
    file_path: str = "../assets/sample.html",  # HTML 文件路径（支持相对/绝对路径）
    open_encoding: str = "utf-8",  # 文件编码
    get_text_separator: str = "\n",  # 提取文本时标签间的拼接分隔符
    bs_kwargs: dict | None = None,  # 传给 BeautifulSoup 的额外参数（如 {"features": "lxml"}）
) -> list[Document]:
    """HTML -> List[Document]

    常见骚操作示例：
    1) 切换解析器：bs_kwargs={"features": "lxml"}
    2) 压缩为单行文本：get_text_separator=" "
    """
    if bs_kwargs is None:
        bs_kwargs = {"features": "html.parser"}

    loader = BSHTMLLoader(
        file_path=file_path,  # 文件路径
        open_encoding=open_encoding,  # 文件编码
        bs_kwargs=bs_kwargs,  # BS4 参数
        get_text_separator=get_text_separator,  # 文本分隔符
    )
    return loader.load()


if __name__ == "__main__":
    print(load_html())
