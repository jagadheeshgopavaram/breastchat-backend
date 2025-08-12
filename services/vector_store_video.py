import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

import os

# ✅ Universal base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
persist_path = os.path.join(BASE_DIR, "data", "vector_storevid")
video_dir = os.path.join(BASE_DIR, "data", "videos")
transcript_dir = os.path.join(BASE_DIR, "data", "transcripts")


def embed_video_transcripts():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    all_chunks = []

    for file in tqdm(os.listdir(transcript_dir), desc="📄 Processing transcripts"):
        if file.endswith(".txt"):
            video_name = file.replace(".txt", "")
            transcript_path = os.path.join(transcript_dir, file)
            video_path = os.path.join(video_dir, f"{video_name}.mp4")

            if not os.path.exists(video_path):
                print(f"❌ Skipping {file}: video not found.")
                continue

            with open(transcript_path, "r", encoding="utf-8") as f:
                content = f.read()

            chunks = text_splitter.split_text(content)
            for chunk in chunks:
                all_chunks.append(
                    Document(
                        page_content=chunk,
                        metadata={"source": video_path.replace("\\", "/")}
                    )
                )

    if not all_chunks:
        print("⚠️ No transcript chunks to embed.")
        return

    print(f"✅ Embedding {len(all_chunks)} transcript chunks from {len(set(doc.metadata['source'] for doc in all_chunks))} videos...")
    db = FAISS.from_documents(all_chunks, embedding=embeddings)
    db.save_local(persist_path)
    print(f"✅ Transcript embeddings saved to: {persist_path}")


def load_video_vector_store():
    """Loads the FAISS vector store containing embedded video transcripts."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    index_file = os.path.join(persist_path, "index.faiss")

    if not os.path.exists(index_file):
        print(f"⚠️ Vector store not found at {index_file}. Skipping load.")
        return None

    return FAISS.load_local(persist_path, embeddings, allow_dangerous_deserialization=True)


def test_query_similarity(query: str, top_k: int = 3):
    db = load_video_vector_store()
    if db is None:
        print("❌ Vector store not loaded. Run embedding first.")
        return

    results = db.similarity_search_with_score(query, k=top_k)

    print(f"\n🔍 Similarity results for query: \"{query}\"\n")
    for i, (doc, score) in enumerate(results):
        print(f"Result {i + 1}:")
        print(f"Text Snippet: {doc.page_content[:100]}...")
        print(f"Video Path: {doc.metadata['source']}")
        print(f"Raw Score: {score:.4f}")
        print("-" * 60)


# ✅ Run manually
if __name__ == "__main__":
    # Step 1: Embed transcripts (only run once unless data changes)
    embed_video_transcripts()

    # Step 2: Test with a query
    test_query_similarity("What is mammogram")
