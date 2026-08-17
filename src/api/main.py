from fastapi import FastAPI
from pydantic import BaseModel

from src.rag import ask


app = FastAPI(
    title="RoadSOS RAG API",
    description="National Road Safety Rules Question Answering API",
    version="1.0.0"
)


# ============================================================
# REQUEST MODEL
# ============================================================

class QuestionRequest(BaseModel):
    question: str


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "status": "online",
        "service": "RoadSOS RAG API"
    }


# ============================================================
# ASK ROAD SAFETY QUESTION
# ============================================================

@app.post("/ask")
def ask_question(request: QuestionRequest):

    result = ask(
        request.question
    )

    return result