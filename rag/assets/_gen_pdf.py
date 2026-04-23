from pypdf import PdfWriter

out_path = "rag/1.loader/assets/sample.pdf"

writer = PdfWriter()
writer.add_blank_page(width=595, height=842)
writer.add_metadata(
    {
        "/Title": "PDF Loader Demo",
        "/Author": "LlangChain",
        "/Subject": "Sample PDF for 1.loader test",
    }
)

with open(out_path, "wb") as f:
    writer.write(f)

print(out_path)
