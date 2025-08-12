from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

def embed_and_store(chunks, persist_path=None):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    texts = [c["content"] for c in chunks]
    metadatas = [{"source": c["source"]} for c in chunks]

    print(f"📦 Total chunks to embed: {len(texts)}")

    db = FAISS.from_texts(texts, embeddings, metadatas=metadatas)

    # Set default persist path
    if persist_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        persist_path = os.path.join(base_dir, "data", "vector_store")

    if not os.path.exists(persist_path):
        os.makedirs(persist_path)

    db.save_local(persist_path)
    print(f"✅ Vector store saved at: {persist_path}")
    return db


def load_vector_store(persist_path=None):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    if persist_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        persist_path = os.path.join(base_dir, "data", "vector_store")

    print(f"📂 Loading vector store from: {persist_path}")
    return FAISS.load_local(persist_path, embeddings, allow_dangerous_deserialization=True)


if __name__ == "__main__":
    from pdf_loader import load_and_split_pdfs

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pdf_folder = os.path.join(base_dir, "data", "pdfs")

    chunks = load_and_split_pdfs(pdf_folder)
    print(f"📑 Total chunks loaded from PDFs: {len(chunks)}")

    db = embed_and_store(chunks)
