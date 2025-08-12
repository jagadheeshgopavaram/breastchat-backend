import sys
import os

# Add parent directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
import google.generativeai as genai
from langchain_core.prompts import PromptTemplate

# --- 📁 Load environment variables ---
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_path = os.path.join(base_dir, "config", "secrets.env")
load_dotenv(env_path)

# --- 🔐 Load Gemini API key ---
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

# --- 🔗 Configure Gemini ---
genai.configure(api_key=api_key)

# --- 🧠 Prompt Templates ---

qa_prompt = PromptTemplate(
    template="""
You are a medical assistant specialized in breast health. Using the context below, answer the question **clearly**, **concisely**, and in **markdown format**.

✅ Guidelines:
- Start with a short summary sentence.
- Use **bold** for key terms.
- Do not use `*`, `_`, or raw HTML.
- Use bullet points for clarity.
- Say "I don't know" if the context doesn’t provide enough information.

---

📚 **Context**:
{context}

❓ **Question**:
{question}
""",
    input_variables=["context", "question"]
)

raghuram_prompt = PromptTemplate(
    template="""
You are Dr. Raghuram’s medical assistant specialized in breast health. Using the context below, answer clearly, concisely, and in **markdown format**.

Context:
{context}

Question:
{question}
""",
    input_variables=["context", "question"]
)

# --- 🔮 Gemini LLM invocation ---
def invoke_llm(prompt: str, model="gemini-2.5-flash", temperature=0.3, max_tokens=2048):
    try:
        model_obj = genai.GenerativeModel(model)
        chat = model_obj.start_chat()
        response = chat.send_message(
            prompt,
            generation_config={
                "temperature": temperature,
                "max_output_tokens": max_tokens
            }
        )
        # Safely extract text
        if hasattr(response, "text") and response.text:
            return response.text
        elif hasattr(response, "candidates") and len(response.candidates) > 0:
            return response.candidates[0].content
        else:
            return "❌ Gemini API returned empty response."
    except Exception as e:
        return f"❌ Error from Gemini API: {str(e)}"

# --- 🧾 Format prompts ---
def format_prompt(context_docs, question, is_raghuram=False):
    context_text = "\n\n".join([doc.page_content for doc in context_docs])
    if is_raghuram:
        return raghuram_prompt.format(context=context_text, question=question)
    else:
        return qa_prompt.format(context=context_text, question=question)

# --- 📺 Video reference ---
from services.vector_store_video import load_video_vector_store

def get_video_reference_link(query: str, top_k: int = 1, score_threshold: float = 0.45) -> str | None:
    try:
        db = load_video_vector_store()
        if db is None:
            return None

        results = db.similarity_search_with_score(query, k=top_k)
        if not results:
            return None

        best_doc, score = results[0]
        if score > score_threshold:
            return None  # Too low relevance

        video_path = best_doc.metadata.get("source", "")
        filename = os.path.basename(video_path)

        return f"http://localhost:3001/{filename}"
    except Exception:
        return None

# --- 🧠 Load Vector Stores ---
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings

RAGHURAM_EMBEDDING_DIR = os.path.join("D:/breasthealth", "data", "embeddings", "raghuram")

GENERAL_EMBEDDING_DIR = os.path.join(base_dir, "data", "vector_store")

# Cache vector stores globally to avoid reloading on every query
_cached_vector_store_raghuram = None
_cached_vector_store_general = None

def load_vector_store_raghuram():
    global _cached_vector_store_raghuram
    if _cached_vector_store_raghuram is None:
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        _cached_vector_store_raghuram = FAISS.load_local(RAGHURAM_EMBEDDING_DIR, embeddings, allow_dangerous_deserialization=True)
    return _cached_vector_store_raghuram

def load_vector_store_general():
    global _cached_vector_store_general
    if _cached_vector_store_general is None:
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        _cached_vector_store_general = FAISS.load_local(GENERAL_EMBEDDING_DIR, embeddings, allow_dangerous_deserialization=True)
    return _cached_vector_store_general

# --- 🧠 Answer with General LLM ---
def answer_question(vector_store, query, k=5):
    try:
        docs = vector_store.similarity_search(query, k=k)
        prompt = format_prompt(docs, query, is_raghuram=False)
        answer_text = invoke_llm(prompt)

        if "i don't know" in answer_text.lower():
            answer_text = "I'm sorry, I don't have enough information to answer that."

        sources = [doc.metadata.get("source", "Unknown") for doc in docs]
        video_link = get_video_reference_link(query)

        return {
            "answer": answer_text,
            "sources": sources,
            "video_link": video_link
        }
    except Exception as e:
        return {
            "answer": f"❌ Error during answer generation: {str(e)}",
            "sources": [],
            "video_link": None
        }

# --- 🧠 Answer with Raghuram LLM ---
def answer_question_raghuram(query, k=5):
    vectorstore = load_vector_store_raghuram()
    docs = vectorstore.similarity_search(query, k=k)
    prompt = format_prompt(docs, query, is_raghuram=True)
    answer = invoke_llm(prompt)
    if "i don't know" in answer.lower():
        answer = "I'm sorry, I don't have enough information to answer that."
    video_link = get_video_reference_link(query)
    return {
        "answer": answer,
        "sources": [],
        "video_link": video_link
    }

# --- 🧪 Raghuram question keyword check ---
def is_raghuram_question_keyword(query: str) -> bool:
    keywords = ["raghuram", "dr. raghuram", "his approach", "raghuram's program"]
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in keywords)

# --- 🧪 LLM classifier to verify Raghuram question ---
def is_raghuram_question_llm(query: str) -> bool:
    classification_prompt = f"""
You are a medical assistant.

Does this question specifically relate to Dr. Raghuram or his approach? Reply only with "Yes" or "No".

Question: "{query}"
"""
    response = invoke_llm(classification_prompt, temperature=0, max_tokens=10)
    return "yes" in response.lower()

# --- 🧪 LLM classifier to check if query relates to breast cancer or Dr. Raghuram ---
def is_query_breast_cancer_related(query: str) -> bool:
    query_lower = query.lower()

    # ✅ Simple keyword check first
    bc_keywords = [
        "breast cancer", "mammogram", "lumpectomy", "mastectomy", "chemotherapy",
        "hormone therapy", "breast lump", "breast screening", "nipple discharge",
        "dr. raghuram", "raghuram"
    ]
    if any(keyword in query_lower for keyword in bc_keywords):
        return True

    # 🤖 LLM-based classification if no keyword match
    classification_prompt = f"""
    You are a medical question classifier.  
    Your ONLY job is to check if a user's question is directly or indirectly related to breast cancer.  
    Answer only "Yes" or "No".

"Related" means:
- Any question about breast cancer symptoms, diagnosis, treatment, screening, risk factors, prevention, side effects, survival rates.
- Questions about mammograms, biopsies, chemotherapy, radiation, mastectomy, breast surgery.
- Questions about breast health, breast lumps, nipple discharge, breast pain.
- Questions about famous doctors or hospitals for breast cancer (e.g., Dr. Raghuram).
- Questions about new research, statistics, awareness campaigns about breast cancer.

"Not related" means:
- Questions about other types of cancer unless directly compared to breast cancer.
- General health topics like diabetes, heart disease, weight loss, unrelated medical advice.
- Personal life advice, tech, politics, unrelated science.


    Question: "{query}"
    """
    response = invoke_llm(classification_prompt, temperature=0, max_tokens=10)
    return "yes" in response.lower()


# --- 🔀 Router function ---
def answer_question_router(query):
    # First try keyword check
    if is_raghuram_question_keyword(query):
        return answer_question_raghuram(query)

    # If keyword fails, try LLM classifier
    if is_raghuram_question_llm(query):
        return answer_question_raghuram(query)

    # Otherwise use general vector store
    vector_store_general = load_vector_store_general()
    return answer_question(vector_store_general, query)


# --- 🧪 CLI test ---
if __name__ == "__main__":
    test_queries = [
        "what is mammogram .",
    ]

    for q in test_queries:
        print(f"\n📥 Query: {q}")
        result = answer_question_router(q)
        print(f"🤖 Answer:\n{result['answer']}")
        print(f"📚 Sources: {result.get('sources')}")
        if result.get("video_link"):
            print(f"🎬 Related video: {result['video_link']}")
        else:
            print("⚠️ No relevant video found.")
