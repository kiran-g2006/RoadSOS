from fastapi import FastAPI
from pydantic import BaseModel

from src.rag import ask


app = FastAPI(
    title="RoadSOS Road Safety RAG API",
    description="RAG API for Indian road safety legislation",
    version="1.0.0"
)


class QuestionRequest(BaseModel):

    question: str


@app.get("/")
def root():

    return {
        "status": "online",
        "service": "RoadSOS Road Safety RAG"
    }


@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


@app.post("/ask")
def ask_question(
    request: QuestionRequest
):

    result = ask(
        request.question
    )

    return result