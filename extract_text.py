import fitz  

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page_num, page in enumerate(doc):
        page_text = page.get_text()
        full_text += page_text
        print(f"--- Page {page_num + 1}: {len(page_text)} characters extracted ---")
    doc.close()
    return full_text

if __name__ == "__main__":
    text = extract_text_from_pdf("data/Test.pdf")
    print("\n=== First 500 characters of extracted text ===")
    print(text[:500])
    print(f"\nTotal characters extracted: {len(text)}")