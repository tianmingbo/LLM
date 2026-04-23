# Markdown 加载示例（长文本版）

这是一个用于测试 **Markdown Loader** 的示例文件，内容刻意写得更长，方便你观察：

- `single` 模式下整篇文档作为一个 `Document`
- `elements` 模式下按结构切分后的块数量和内容质量
- 检索时标题、表格、代码片段、引用是否能被有效召回

## 1. 背景

在 RAG 场景里，Markdown 是很常见的知识载体，比如团队文档、技术方案、接口说明、运维手册、FAQ 等。  
如果只做“整篇读入”，召回粒度会比较粗；如果按结构切分（标题、段落、列表、表格）再向量化，通常检索命中率会更稳定。

## 2. 快速示例

下面是一段最简单的示例代码，用于演示从 Markdown 文件加载内容：

```python
from rag.loader.load_markdown import load_markdown

docs = load_markdown(
    file_path="rag/1.loader/assets/sample.md",
    mode="single",
)
print(len(docs), docs[0].metadata)
```

如果你想按块解析：

```python
docs = load_markdown(
    file_path="rag/1.loader/assets/sample.md",
    mode="elements",
)
for i, d in enumerate(docs[:5], 1):
    print(i, d.metadata)
```

## 3. 功能清单

### 3.1 加载策略

- 单文档加载：适合小文件或快速验证
- 结构化加载：适合生产场景的检索
- 回退加载：当 `unstructured` 依赖不可用时自动回退 `TextLoader`

### 3.2 元数据建议

- `source`：原始文件路径
- `section`：一级或二级标题
- `chunk_id`：切分后的块编号
- `updated_at`：文档更新时间（便于增量更新）

## 4. 参数对照表

| 参数名 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `file_path` | `str` | `rag/loader/assets/sample.md` | Markdown 文件路径 |
| `mode` | `str` | `single` | `single/elements` |
| `encoding` | `str` | `utf-8` | 文件编码 |
| `autodetect_encoding` | `bool` | `False` | 是否自动探测编码 |
| `use_unstructured` | `bool` | `True` | 是否优先使用 Unstructured |
| `fallback_to_text` | `bool` | `True` | 失败时是否回退 TextLoader |

## 5. 实战建议

1. 文档很长时，不建议只依赖单一 chunk 策略。  
2. 先按结构切块，再按 token 长度二次切块，通常效果更好。  
3. 对代码块和表格可以单独处理，避免被普通段落切分器破坏语义。  
4. 对多语言文档，建议在 metadata 中记录 `language` 字段。  

## 6. 代码块（多语言）

```python
def retrieve(query: str) -> list[str]:
    # 示例：向量检索 + rerank
    return ["chunk-1", "chunk-2"]
```

```sql
SELECT id, title, score
FROM doc_chunks
WHERE dataset = 'kb_markdown'
ORDER BY score DESC
LIMIT 5;
```

```bash
source .venv/bin/activate
python3 rag/1.loader/load_markdown.py
```

## 7. 引用与说明

> 结构化文本在检索增强生成（RAG）中通常优于纯大段文本，  
> 因为它能更好地保留语义边界并降低噪声召回概率。

补充说明：上面这句话是工程经验总结，不代表某个固定 benchmark 的绝对结论，实际效果仍依赖你的数据分布、切分策略、embedding 模型和 rerank 策略。

## 8. 结尾

这份 Markdown 样例包含了标题、段落、列表、表格、引用、多代码块等常见结构，足够用于测试 loader、splitter 和向量检索链路。
