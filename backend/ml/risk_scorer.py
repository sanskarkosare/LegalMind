from langchain_groq import ChatGroq
from pypdf import PdfReader
from dotenv import load_dotenv
import os
import json

load_dotenv()

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="groq/compound-mini",
    temperature=0.2
)

def extract_text_from_pdf(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text.strip()

def extract_text_from_txt(txt_path: str) -> str:
    with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()

def summarize_contract(text: str) -> dict:
    prompt = f"""You are a legal AI assistant. Analyze this contract and provide:

1. SUMMARY: 2-3 sentence plain English summary
2. PARTIES: Who are the parties involved
3. KEY_TERMS: List 5 most important terms
4. RED_FLAGS: Any concerning or unusual clauses
5. RECOMMENDATIONS: What to negotiate or be careful about

Contract text:
{text[:3000]}

Respond ONLY in this JSON format:
{{
  "summary": "...",
  "parties": "...",
  "key_terms": ["term 1", "term 2", "term 3", "term 4", "term 5"],
  "red_flags": ["flag 1", "flag 2"],
  "recommendations": ["rec 1", "rec 2", "rec 3"]
}}"""

    response = llm.invoke([{"role": "user", "content": prompt}])

    try:
        content = response.content.strip()
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        return json.loads(content.strip())
    except:
        return {
            "summary": response.content[:500],
            "parties": "Could not extract",
            "key_terms": [],
            "red_flags": [],
            "recommendations": []
        }

def answer_legal_question(question: str, context: str = "") -> str:
    system = """You are LegalMind, an AI legal assistant.
Provide clear, accurate legal information in simple language.
Always remind users to consult a qualified lawyer for legal advice.
Be concise and helpful."""

    context_line = f"Contract context: {context[:500]}" if context else ""

    response = llm.invoke([
        {"role": "system", "content": f"{system}\n{context_line}"},
        {"role": "user", "content": question}
    ])
    return response.content