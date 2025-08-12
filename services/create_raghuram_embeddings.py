import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS  # or Chroma if you prefer
from langchain.embeddings import SentenceTransformerEmbeddings  # free local embeddings

# Paths
PDF_DIR = "data/pdfs/raghuram"
EMBEDDING_DIR = "data/embeddings/raghuram"

os.makedirs(EMBEDDING_DIR, exist_ok=True)

# Load PDFs and split text
def load_and_split_pdfs(pdf_dir):
    docs = []
    for filename in os.listdir(pdf_dir):
        if filename.lower().endswith(".pdf"):
            path = os.path.join(pdf_dir, filename)
            loader = PyPDFLoader(path)
            pdf_docs = loader.load()
            docs.extend(pdf_docs)
            print(f"Loaded and parsed: {filename} ({len(pdf_docs)} pages)")
    return docs

# Split documents into chunks (e.g., 1000 chars with 200 overlap)
def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(docs)
    print(f"Split into {len(split_docs)} chunks")
    return split_docs

def create_and_save_vectorstore(docs, embedding_dir):
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(embedding_dir)
    print(f"Vector store saved at: {embedding_dir}")

if __name__ == "__main__":
    documents = load_and_split_pdfs(PDF_DIR)
    chunks = split_docs(documents)
    create_and_save_vectorstore(chunks, EMBEDDING_DIR)
