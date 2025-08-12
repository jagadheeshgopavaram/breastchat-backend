import os
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_split_pdfs(pdf_folder):
    all_chunks = []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]
    print(f"📚 Found {len(pdf_files)} PDF(s) in '{pdf_folder}'")

    for filename in pdf_files:
        pdf_path = os.path.join(pdf_folder, filename)
        doc = fitz.open(pdf_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text()
        doc.close()

        chunks = text_splitter.split_text(full_text)
        print(f"📄 {filename} → {len(chunks)} chunks")

        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "source": filename,
                "content": chunk
            })

    print(f"✅ Total chunks created from all PDFs: {len(all_chunks)}")
    return all_chunks

# ✅ CLI test
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pdf_folder = os.path.join(base_dir, "data", "pdfs")
    chunks = load_and_split_pdfs(pdf_folder)

    print(f"\n🔍 Showing sample chunks:")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n--- Chunk {i+1} ---")
        print(f"Source: {chunk['source']}")
        print(chunk["content"][:300])
