import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path: str) -> list[str]:
    """
    Extracts text by pages and returns list of page texts.
    """
    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        pages.append(page.get_text())
    return pages

def chunk_text(pages: list[str], max_chunk_size=500) -> list[str]:
    """
    Simple chunking: split page text into chunks of max_chunk_size words.
    """
    chunks = []
    for page_text in pages:
        words = page_text.split()
        for i in range(0, len(words), max_chunk_size):
            chunk = " ".join(words[i:i+max_chunk_size])
            chunks.append(chunk)
    return chunks
