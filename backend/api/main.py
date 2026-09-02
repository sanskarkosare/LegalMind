import os
import shutil
import tempfile
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from backend.ml.clause_classifier import analyze_contract
from backend.ml.risk_scorer import extract_text_from_pdf, summarize_contract, answer_legal_question

app = FastAPI(title="LegalMind API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

class QuestionRequest(BaseModel):
    question: str
    context: str = ""

@app.get("/")
async def root():
    return {"status": "LegalMind API is Online"}

@app.post("/analyze-contract")
async def analyze(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    text = extract_text_from_pdf(tmp_path)
    os.remove(tmp_path)
    if not text:
        return {"error": "Could not extract text from PDF."}
    clause_analysis = analyze_contract(text)
    summary = summarize_contract(text)
    return {
        "clause_analysis": clause_analysis,
        "summary": summary,
        "text_length": len(text)
    }

@app.post("/ask")
async def ask(request: QuestionRequest):
    answer = answer_legal_question(request.question, request.context)
    return {"answer": answer}