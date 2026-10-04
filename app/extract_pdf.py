import fitz

pdf_path = "data/sample.pdf"

doc = fitz.open(pdf_path)

print("Total pages:", len(doc))

for page_number, page in enumerate(doc):
    text = page.get_text()

    print("\n==============================")
    print("PAGE", page_number + 1)
    print("==============================")

    print(text)