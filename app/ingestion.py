from pypdf import PdfReader
import os


def load_pdf(path):
    reader = PdfReader(path)
    text = ""

    for page in reader.pages:
        text += page.extract_text() or ""

    return text


def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks


def load_documents(folder_path):
    all_chunks = []

    for filename in os.listdir(folder_path):

        if not filename.endswith(".pdf"):
            continue

        file_path = os.path.join(folder_path, filename)

        text = load_pdf(file_path)

        chunks = chunk_text(text)

        for chunk in chunks:
            all_chunks.append({
                "text": chunk,
                "source": filename
            })

    return all_chunks