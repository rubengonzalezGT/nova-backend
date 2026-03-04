import re
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text
from pydantic import BaseModel
from typing import Optional

from app.core.database import get_db
from app.core.security import get_current_user
from app.models.user import User

router = APIRouter(prefix="/pdf", tags=["PDF Ask"])

STOPWORDS = {
    "cual", "cuales", "como", "donde", "cuando", "quie", "quien",
    "quienes", "para", "segun", "sobre", "entre", "esta", "esto",
    "este", "estan", "tiene", "tienen", "puede", "pueden", "debe",
    "deben", "cual", "cuales", "como", "donde", "cuando", "quien",
    "quienes", "segun", "estan", "que", "son", "las", "los", "del",
    "una", "uno", "unos", "unas", "por", "con", "sin", "hay",
}


class PdfAskResponse(BaseModel):
    question: str
    found: bool
    answer: Optional[str]
    source_file: Optional[str]
    page_hint: Optional[str]
    chunks_found: int


def extract_best_sentences(chunk: str, keywords: list[str], context: int = 0) -> str:
    text = re.sub(r'\s+', ' ', chunk.strip())
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

    if not sentences:
        return text[:500]

    hits = []
    for i, sentence in enumerate(sentences):
        s = sentence.lower()
        s_clean = re.sub(r'[áàä]', 'a', re.sub(r'[éèë]', 'e', re.sub(r'[íìï]', 'i',
                  re.sub(r'[óòö]', 'o', re.sub(r'[úùü]', 'u', s)))))
        if any(kw in s_clean for kw in keywords):
            hits.append(i)

    if not hits:
        return " ".join(sentences[:2])

    best_hits = hits[:2]
    indices = set()
    for h in best_hits:
        for i in range(max(0, h - context), min(len(sentences), h + context + 1)):
            indices.add(i)

    return " ".join(sentences[i] for i in sorted(indices))


def format_answer(question: str, answer: str) -> str:
    q = question.lower().strip()

    if q.startswith("que es") or q.startswith("qué es"):
        tema = question.split(" ", 2)[-1].strip().capitalize()
        return f"{tema} es: {answer}"

    elif q.startswith("cuales son") or q.startswith("cuáles son"):
        tema = question.split(" ", 2)[-1].strip().capitalize()
        return f"Las/Los {tema} son: {answer}"

    elif q.startswith("como") or q.startswith("cómo"):
        tema = question.split(" ", 1)[-1].strip().capitalize()
        return f"{tema}: {answer}"

    elif q.startswith("quien") or q.startswith("quién"):
        tema = question.split(" ", 1)[-1].strip().capitalize()
        return f"{tema}: {answer}"

    elif q.startswith("donde") or q.startswith("dónde"):
        tema = question.split(" ", 1)[-1].strip().capitalize()
        return f"Ubicación de {tema}: {answer}"

    elif q.startswith("cuando") or q.startswith("cuándo"):
        tema = question.split(" ", 1)[-1].strip().capitalize()
        return f"Fecha/Momento de {tema}: {answer}"

    else:
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

    keywords = [
        w for w in question.lower().split()
        if len(w) > 3 and w not in STOPWORDS
    ]

    if not keywords:
        raise HTTPException(status_code=400, detail="La pregunta es muy corta o solo contiene palabras comunes.")

    best_keyword = max(keywords, key=len)
    like_q = "%" + best_keyword + "%"

    rows = db.execute(text("""
        SELECT e.chunk_text, e.chunk_index, k.title AS filename
        FROM embeddings e
        JOIN knowledge_items k ON k.id = e.knowledge_id
        WHERE k.source = 'pdf'
          AND unaccent(LOWER(e.chunk_text)) LIKE unaccent(LOWER(:like_q))
        ORDER BY e.chunk_index ASC
        LIMIT 5
    """), {"like_q": like_q}).fetchall()

    if not rows:
        return PdfAskResponse(
            question=question, found=False,
            answer=None, source_file=None,
            page_hint=None, chunks_found=0,
        )

    best = rows[0]
    answer = extract_best_sentences(best.chunk_text, keywords, context=0)
    answer = format_answer(question, answer)

    return PdfAskResponse(
        question=question, found=True,
        answer=answer,
        source_file=best.filename,
        page_hint=f"Chunk #{best.chunk_index + 1}",
        chunks_found=len(rows),
    )