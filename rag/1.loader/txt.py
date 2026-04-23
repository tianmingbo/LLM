# pip install langchain_community
from langchain_community.document_loaders import TextLoader

docs = TextLoader(
    file_path="../assets/sample.txt",
    encoding="utf-8",
).load()

print(docs)
