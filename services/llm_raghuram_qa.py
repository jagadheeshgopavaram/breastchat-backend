import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.prompts import PromptTemplate
import google.generativeai as genai

# Embedding directory path
EMBEDDING_DIR = "data/embeddings/raghuram"

# Load Gemini API key from environment

from dotenv import load_dotenv

# Explicitly load your .env from the config folder
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "secrets.env")
load_dotenv(env_path)

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")


# Configure Gemini API
genai.configure(api_key=api_key)

# Load vectorstore with local embeddings
def load_vectorstore():
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(EMBEDDING_DIR, embeddings, allow_dangerous_deserialization=True)
    return vectorstore


# Prompt template for QA
qa_prompt = PromptTemplate(
    template="""
You are Dr. Raghuram’s medical assistant specialized in breast health. Using the context below, answer the question clearly, concisely, and in **markdown**.

✅ Guidelines:
- Start with a short summary sentence.
- Use **bold** for key terms.
- Use bullet points for clarity.
- Avoid raw HTML or special markdown characters like `*` or `_`.
- If the context doesn’t provide enough information, say "I don't know".

Context:
{context}

Question:
{question}
""",
    input_variables=["context", "question"]
)

# Call Gemini generative model API with prompt
def invoke_llm(prompt: str, model="gemini-2.5-flash", temperature=0.3, max_tokens=1024):
    model_obj = genai.GenerativeModel(model)
    chat = model_obj.start_chat()
    response = chat.send_message(
        prompt,
        generation_config={
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
    )
    return response.text

# Format the prompt with document context + question
def format_prompt(context_docs, question):
    context_text = "\n\n".join([doc.page_content for doc in context_docs])
    return qa_prompt.format(context=context_text, question=question)

# Main function to answer question from PDF embeddings + Gemini
def answer_question(question: str, k=5):
    vectorstore = load_vectorstore()
    docs = vectorstore.similarity_search(question, k=k)
    prompt = format_prompt(docs, question)
    answer = invoke_llm(prompt)
    return answer

# CLI test example

