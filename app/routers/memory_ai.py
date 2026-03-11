import re
import unicodedata
import uuid
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import func, or_

from app.models.user import KnowledgeItem, KnowledgeSource
from app.core.database import get_db
from app.core.security import get_current_user
from app.models.user import User, QaMemory, PdfChunk
from sqlalchemy import text
from app.schemas.schemas import (
    MemoryLearnRequest,
    MemoryLearnResponse,
    MemoryCandidate,
    MemoryStatsResponse,
)

router = APIRouter(prefix="/memory", tags=["Memory AI"])

SYSTEM_EMAIL = "system@nova.com"


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def get_global_user(db: Session) -> User:
    user = db.query(User).filter(User.email == SYSTEM_EMAIL).first()
    if not user:
        raise HTTPException(status_code=500, detail="Usuario global no encontrado")
    return user


def normalize_question(q: str) -> str:
    q = q.strip().lower()
    q = unicodedata.normalize("NFKD", q)
    q = "".join(c for c in q if not unicodedata.combining(c))
    q = re.sub(r"[^a-z0-9\s]", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q


def normalize_text(text: str) -> str:
    text = text.lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return text


STOPWORDS = {
    "que", "qué", "como", "cómo", "cual", "cuál", "cuales", "cuáles",
    "de", "del", "la", "las", "el", "los", "y", "o", "en", "por",
    "para", "con", "sin", "un", "una", "unos", "unas", "es", "son",
    "se", "lo", "le", "les", "a", "al"
}


def extract_keywords(question: str):
    normalized = normalize_text(question)
    words = re.findall(r"[a-z0-9]+", normalized)
    keywords = [w for w in words if len(w) > 3 and w not in STOPWORDS]
    return [w[:6] for w in keywords]


# ─────────────────────────────────────────────────────────────
# Buscar memoria manual
# ─────────────────────────────────────────────────────────────

def search_memory(db: Session, uid: str, q: str):
    return db.execute(text("""
        SELECT question, answer, votes, similarity(question, :q) AS sim
        FROM qa_memory
        WHERE user_id = :uid AND question % :q
        ORDER BY sim DESC, votes DESC, updated_at DESC
        LIMIT 20
    """), {"q": q, "uid": uid}).fetchall()


# ─────────────────────────────────────────────────────────────
# Buscar en PDFs
# ─────────────────────────────────────────────────────────────

def search_pdf_chunks(db: Session, question: str):

    keywords = extract_keywords(question)

    if not keywords:
        return []

    filters = [
        PdfChunk.chunk_text.ilike(f"%{kw}%")
        for kw in keywords[:5]
    ]

    return (
        db.query(PdfChunk)
        .filter(or_(*filters))
        .order_by(PdfChunk.chunk_index.asc())
        .limit(5)
        .all()
    )


def extract_pdf_answer(chunk_text: str, question: str):

    keywords = extract_keywords(question)

    text = re.sub(r"\s+", " ", chunk_text).strip()

    sentences = [
        s.strip()
        for s in re.split(r'(?<=[.!?])\s+', text)
        if len(s.strip()) > 20
    ]

    if not sentences:
        return text[:500]

    normalized_sentences = [normalize_text(s) for s in sentences]

    hits = []

    for i, sentence in enumerate(normalized_sentences):

        score = sum(1 for kw in keywords if kw in sentence)

        if score > 0:
            hits.append((i, score))

    if not hits:
        return " ".join(sentences[:2])[:500]

    hits.sort(key=lambda x: x[1], reverse=True)

    best_index = hits[0][0]

    start = max(0, best_index - 1)
    end = min(len(sentences), best_index + 2)

    result = " ".join(sentences[start:end])

    return result[:500]


# ─────────────────────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str


class TeachRequest(BaseModel):
    question: str
    answer: str


# ─────────────────────────────────────────────────────────────
# Ask (CHAT)
# ─────────────────────────────────────────────────────────────

@router.post("/ask")
def ask(
    data: AskRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):

    system_user = get_global_user(db)

    q = normalize_question(data.question)

    # 1️⃣ buscar en memoria manual

    rows = search_memory(db, str(system_user.id), q)

    if rows:

        scores = {}

        for r in rows:
            scores[r.answer] = scores.get(r.answer, 0.0) + float(r.sim) * int(r.votes)

        best_answer, _ = max(scores.items(), key=lambda x: x[1])

        top_sim = float(rows[0].sim)

        if top_sim >= 0.35:

            return {
                "knows": True,
                "answer": best_answer,
                "source": "memory"
            }

    # 2️⃣ buscar en PDFs

    pdf_chunks = search_pdf_chunks(db, data.question)

    if pdf_chunks:

        best_chunk = pdf_chunks[0]

        pdf_answer = extract_pdf_answer(best_chunk.chunk_text, data.question)

        return {
            "knows": True,
            "answer": pdf_answer,
            "source": "pdf",
            "file": best_chunk.filename,
            "chunk": best_chunk.chunk_index + 1
        }

    # 3️⃣ no sabe

    return {
        "knows": False,
        "message": "No tengo información sobre eso todavía. ¿Me puedes enseñar?"
    }


# ─────────────────────────────────────────────────────────────
# Teach
# ─────────────────────────────────────────────────────────────

@router.post("/teach")
def teach(
    data: TeachRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):

    system_user = get_global_user(db)

    q = normalize_question(data.question)

    a = data.answer.strip()

    if not q:
        raise HTTPException(status_code=422, detail="La pregunta no puede estar vacía")

    if not a:
        raise HTTPException(status_code=422, detail="La respuesta no puede estar vacía")

    existing = db.query(QaMemory).filter(
        QaMemory.user_id == system_user.id,
        QaMemory.question == q,
        QaMemory.answer == a
    ).first()

    if existing:

        existing.votes += 1

        existing.updated_at = func.now()

    else:

        db.add(QaMemory(
            user_id=system_user.id,
            question=q,
            answer=a,
            votes=1
        ))

    db.commit()
    # Guardar en knowledge_items para el calendario

    knowledge = KnowledgeItem(
        id=uuid.uuid4(),
        created_by=current_user.id,
        title=data.question,
        content=a,
        source=KnowledgeSource.manual,
        tags=[]
    )
    db.add(knowledge)
    db.commit()

    return {
        "question": data.question,
        "answer": a,
        "message": "Aprendido correctamente"
    }


# ─────────────────────────────────────────────────────────────
# Stats
# ─────────────────────────────────────────────────────────────

@router.get("/stats", response_model=MemoryStatsResponse)
def stats(
    question: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):

    system_user = get_global_user(db)

    q = normalize_question(question)

    rows = search_memory(db, str(system_user.id), q)

    if not rows:

        return MemoryStatsResponse(
            question_original=question,
            question_normalized=q,
            total_candidates=0,
            chosen_answer=None,
            candidates=[]
        )

    scores = {}

    candidates = []

    for r in rows:

        score = float(r.sim) * int(r.votes)

        scores[r.answer] = scores.get(r.answer, 0.0) + score

        candidates.append(MemoryCandidate(
            answer=r.answer,
            votes=r.votes,
            similarity=round(float(r.sim), 4),
            score=round(score, 4)
        ))

    candidates.sort(key=lambda x: x.score, reverse=True)

    top_sim = float(rows[0].sim)

    chosen = max(scores.items(), key=lambda x: x[1])[0] if top_sim >= 0.35 else None

    return MemoryStatsResponse(
        question_original=question,
        question_normalized=q,
        total_candidates=len(candidates),
        chosen_answer=chosen,
        candidates=candidates
    )