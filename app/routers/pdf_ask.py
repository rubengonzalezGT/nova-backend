import re
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional

from app.core.database import get_db
from app.core.security import get_current_user
from app.models.user import User, PdfChunk

router = APIRouter(prefix="/pdf", tags=["PDF Ask"])

STOPWORDS = {
    "cual", "cuales", "como", "donde", "cuando", "quien", "quienes",
    "para", "segun", "sobre", "entre", "esta", "esto", "este", "estan",
    "tiene", "tienen", "puede", "pueden", "debe", "deben", "que", "son",
    "las", "los", "del", "una", "uno", "unos", "unas", "por", "con", "sin", "hay",
}


class PdfAskResponse(BaseModel):
    question: str
    found: bool
    answer: Optional[str]
    source_file: Optional[str]
    page_hint: Optional[str]
    chunks_found: int


def clean_chunk(text: str) -> str:
    text = re.sub(r'[~\-]\s*\d+\s*[~\-]', ' ', text)
    lines = [l for l in text.split('\n') if len(l.strip()) > 15]
    text = re.sub(r'\s+', ' ', ' '.join(lines)).strip()
    return text


def extract_best_sentences(chunk: str, keywords: list, context: int = 1) -> str:
    chunk = clean_chunk(chunk)
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', chunk) if len(s.strip()) > 20]

    if not sentences:
        return chunk[:400]

    def normalize(s):
        for a, b in [('á','a'),('é','e'),('í','i'),('ó','o'),('ú','u')]:
            s = s.replace(a, b)
        return s.lower()

    hits = [i for i, s in enumerate(sentences) if any(kw in normalize(s) for kw in keywords)]

    if not hits:
        return " ".join(sentences[:2])

    indices = set()
    for h in hits[:2]:
        for i in range(max(0, h - context), min(len(sentences), h + context + 1)):
            indices.add(i)

    result = " ".join(sentences[i] for i in sorted(indices))
    return result[:500].rsplit(' ', 1)[0] + "..." if len(result) > 500 else result


def format_answer(question: str, answer: str) -> str:
    q = question.lower().strip()
    if q.startswith(("que es", "qué es")):
        return f"{question.split(' ', 2)[-1].strip().capitalize()} es: {answer}"
    elif q.startswith(("cuales son", "cuáles son")):
        return f"Las/Los {question.split(' ', 2)[-1].strip().capitalize()} son: {answer}"
    elif q.startswith(("como", "cómo", "quien", "quién")):
        return f"{question.split(' ', 1)[-1].strip().capitalize()}: {answer}"
    return answer


@router.get("/ask", response_model=PdfAskResponse)
def pdf_ask(
    question: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    question = question.replace("?", "").replace("¿", "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="La pregunta no puede estar vacía.")

    keywords = [w for w in question.lower().split() if len(w) > 3 and w not in STOPWORDS]
    if not keywords:
        raise HTTPException(status_code=400, detail="La pregunta es muy corta.")

    best_keyword = max(keywords, key=len)

    # ORM con ilike en lugar de SQL raw
    chunks = db.query(PdfChunk).filter(
        PdfChunk.chunk_text.ilike(f"%{best_keyword}%")
    ).order_by(PdfChunk.chunk_index.asc()).limit(5).all()

    if not chunks:
        return PdfAskResponse(
            question=question, found=False,
            answer=None, source_file=None,
            page_hint=None, chunks_found=0,
        )

    best = chunks[0]
    answer = extract_best_sentences(best.chunk_text, keywords)
    answer = format_answer(question, answer)

    return PdfAskResponse(
        question=question, found=True,
        answer=answer,
        source_file=best.filename,
        page_hint=f"Chunk #{best.chunk_index + 1}",
        chunks_found=len(chunks),
    )