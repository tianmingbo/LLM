from langchain_community.document_loaders import JSONLoader
from langchain_core.documents import Document


def _metadata_func(record: dict, metadata: dict) -> dict:
    metadata["id"] = record.get("id")
    metadata["title"] = record.get("title")
    return metadata


def load_json(
    file_path: str = "../assets/sample.json",  # JSON 文件路径（支持相对/绝对路径）
    jq_schema: str = ".[]",  # jq 选择器：按条遍历数组用 .[]，取嵌套字段可用 .data.items[]
    content_key: str = "content",  # 每条记录中作为 page_content 的字段名
    is_content_key_jq_parsable: bool = False,  # content_key 是否也按 jq 表达式解析
    text_content: bool = True,  # True: 仅保留文本；False: 保留 JSON 字符串内容
    json_lines: bool = False,  # 是否按 JSONL（每行一个 JSON）读取
) -> list[Document]:
    """JSON -> List[Document]

    常见骚操作示例：
    1) 取嵌套数组：jq_schema=".payload.records[]"
    2) 动态拼内容：content_key=".title + \" | \" + .content", is_content_key_jq_parsable=True
    3) 读取 JSONL：json_lines=True
    """
    loader = JSONLoader(
        file_path=file_path,  # 文件路径
        jq_schema=jq_schema,  # 记录选择规则
        content_key=content_key,  # 内容字段/表达式
        is_content_key_jq_parsable=is_content_key_jq_parsable,  # 是否启用 jq content_key
        text_content=text_content,  # 文本模式
        json_lines=json_lines,  # JSONL 模式
        metadata_func=_metadata_func,  # 自定义元数据映射
    )
    return loader.load()


if __name__ == "__main__":
    print(load_json())
