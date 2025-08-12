import logging
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from services.llm_qa import answer_question_router, is_query_breast_cancer_related

app = FastAPI()

# Setup logging for debug info
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CORS setup for frontend URLs
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files route for videos
videos_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "breastchat-ui", "public")
)
app.mount("/videos", StaticFiles(directory=videos_dir), name="videos")

class Query(BaseModel):
    question: str

@app.get("/")
def read_root():
    return {"message": "Backend is working!"}

@app.post("/api/chat")
async def chat(query: Query):
    question = query.question
    logger.info(f"Received question: {question}")

    related = is_query_breast_cancer_related(question)
    logger.info(f"Is question breast cancer related? {related}")

    result = answer_question_router(question)
    logger.info(f"Result from answer_question_router: {result}")

    video_link = None
    if result.get("video_link"):
        filename = os.path.basename(result["video_link"])
        video_link = f"http://localhost:3001/{filename}"
        logger.info(f"Video link in result: {result['video_link']}")
        logger.info(f"Final video_link sent to frontend: {video_link}")
    else:
        logger.info("No video link found in result.")

    return {
        "answer": result.get("answer", "Sorry, no answer found."),
        "sources": result.get("sources", []),
        "video_link": video_link,
    }
