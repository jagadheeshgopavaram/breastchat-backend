import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# At the top of the script
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
persist_path = os.path.join(BASE_DIR, "data", "vector_storevid")
video_dir = os.path.join(BASE_DIR, "data", "videos")
transcript_dir = os.path.join(BASE_DIR, "data", "transcripts")

embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"


def embed_video_transcripts():
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    all_chunks = []

    for file in os.listdir(transcript_dir):
        if file.endswith(".txt"):
            video_name = file.replace(".txt", "")
            transcript_path = os.path.join(transcript_dir, file)
            video_path = os.path.join(video_dir, f"{video_name}.mp4")

            # ✅ Only embed if video file exists
            if not os.path.exists(video_path):
                print(f"❌ Skipping {file}: video not found.")
                continue

            with open(transcript_path, "r", encoding="utf-8") as f:
                content = f.read()

            chunks = text_splitter.split_text(content)
            for chunk in chunks:
                all_chunks.append(
                    Document(page_content=chunk, metadata={"source": video_path.replace("\\", "/")})
                )

    if not all_chunks:
        print("⚠️ No transcript chunks to embed.")
        return

    print(f"✅ Embedding {len(all_chunks)} transcript chunks...")
    db = FAISS.from_documents(all_chunks, embedding=embeddings)
    db.save_local(persist_path)
    print(f"✅ Transcript embeddings saved to {persist_path}")


def verify_embedding_match(faiss_path, embeddings):
    import faiss
    index_file = os.path.join(faiss_path, "index.faiss")
    if not os.path.exists(index_file):
        return True  # First-time load is fine

    index = faiss.read_index(index_file)
    model_dim = len(embeddings.embed_query("test"))
    if index.d != model_dim:
        print(f"❌ FAISS index dim = {index.d}, but embedding model returns dim = {model_dim}")
        print(f"🧹 Please delete: {faiss_path}\\index.faiss and index.pkl, then re-run embedding.")
        return False
    return True


def load_video_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    index_file = os.path.join(persist_path, "index.faiss")

    if not os.path.exists(index_file):
        print(f"⚠️ Vector store not found at {index_file}.")
        return None

    if not verify_embedding_match(persist_path, embeddings):
        return None

    return FAISS.load_local(persist_path, embeddings, allow_dangerous_deserialization=True)


# ✅ Manual run
if __name__ == "__main__":
    embed_video_transcripts()
