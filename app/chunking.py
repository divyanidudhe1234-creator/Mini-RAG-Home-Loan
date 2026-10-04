import fitz

pdf_path = "data/sample.pdf"

doc = fitz.open(pdf_path)

chunk_size = 500
overlap = 100

chunks = []

for page_number, page in enumerate(doc):
    text = page.get_text().strip()

    start = 0

    while start < len(text):
        end = start + chunk_size

        chunk_text = text[start:end]

        chunks.append({
            "text": chunk_text,
            "page": page_number + 1,
            "source": "sample.pdf"
        })

        start = end - overlap

print("Total chunks:", len(chunks))

for i, chunk in enumerate(chunks[:5]):
    print("\n-----------------------------")
    print("CHUNK", i + 1)
    print("Page:", chunk["page"])
    print("Source:", chunk["source"])
    print(chunk["text"])