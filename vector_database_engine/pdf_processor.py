import os
import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter


def extract_text_from_pdf(pdf_path: str) -> str:
    """Open the PDF and return all its text as one big string."""
    doc = pymupdf.open(pdf_path)
    full_text = ""
    for page in doc:                       # go page by page
        full_text += page.get_text() + "\n"
    doc.close()
    return full_text


def split_text_into_chunks(text: str) -> list:
    """Cut the big text into 1000-character chunks with 200 overlap."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    return splitter.split_text(text)


def process_pdf(pdf_path: str) -> list:
    """Master function: PDF file in, list of chunks out."""
    text = extract_text_from_pdf(pdf_path)
    return split_text_into_chunks(text)


# This block only runs when you run THIS file directly (a quick test)
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(base_dir, "sample.pdf")

    doc = pymupdf.open(pdf_path)
    print("Pages in PDF:", len(doc))
    doc.close()

    text = extract_text_from_pdf(pdf_path)
    print("Characters of text found:", len(text.strip()))

    if not text.strip():
        print("\nNo text found. This PDF is probably made of images.")
        print("Replace sample.pdf with a PDF where you can select the text.")
    else:
        chunks = split_text_into_chunks(text)
        print("Total chunks:", len(chunks))
        print("\n--- First chunk ---\n")
        print(chunks[0])