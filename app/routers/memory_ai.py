import re
import unicodedata
import uuid
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.core.database import get_db
from app.core.security import get_current_user
from app.models.user import User, QaMemory
from app.schemas.schemas import (
    MemoryLearnRequest, MemoryLearnResponse,
    MemoryCandidate, MemoryStatsResponse,
)
from app.models.user import KnowledgeItem, KnowledgeSource
from app.services.embedding_service import get_embedding

router = APIRouter(prefix="/memory", tags=["Memory AI"])

SYSTEM_EMAIL = "system@nova.com"


# ── Helpers ───────────────────────────────────────────────────

def get_global_user(db: Session) -> User:
    user = db.query(User).filter(User.email == SYSTEM_EMAIL).first()
    if not user:
        raise HTTPException(status_code=500, detail=f"No existe usuario global: {SYSTEM_EMAIL}")
    return user


def normalize_question(q: str) -> str:
    q = q.strip().lower()
    q = unicodedata.normalize("NFKD", q)
    q = "".join(c for c in q if not unicodedata.combining(c))
    q = re.sub(r"[^a-z0-9\s]", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q


def search_memory(db: Session, uid: str, q: str):
    """Busca en qa_memory usando pg_trgm similarity."""
    return db.execute(
        __import__('sqlalchemy').text("""
            SELECT question, answer, votes, similarity(question, :q) AS sim
            FROM qa_memory
            WHERE user_id = :uid AND question % :q
            ORDER BY sim DESC, votes DESC, updated_at DESC
            LIMIT 20
        """),
        {"q": q, "uid": uid}
    ).fetchall()


# ── Schemas locales ───────────────────────────────────────────

class AskRequest(BaseModel):
    question: str

class TeachRequest(BaseModel):
    question: str
    answer: str


# ── Endpoints ─────────────────────────────────────────────────

@router.post("/ask")
def ask(
    data: AskRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    system_user = get_global_user(db)
    q = normalize_question(data.question)
    rows = search_memory(db, str(system_user.id), q)

    if not rows:
        return {"knows": False, "message": "No sé la respuesta. ¿Me puedes enseñar?"}

    scores = {}
    for r in rows:
        scores[r.answer] = scores.get(r.answer, 0.0) + float(r.sim) * int(r.votes)

    best_answer, _ = max(scores.items(), key=lambda x: x[1])
    top_sim = float(rows[0].sim)

    if top_sim < 0.35:
        return {"knows": False, "message": "No sé la respuesta a tu pregunta. ¿Me puedes ayudar diciendo lo que entiendes sobre tu pregunta?"}

    return {"knows": True, "answer": best_answer, "message": f"Similitud top={top_sim:.2f}."}


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

    # Guardar o reforzar en qa_memory
    existing = db.query(QaMemory).filter(
        QaMemory.user_id == system_user.id,
        QaMemory.question == q,
        QaMemory.answer == a,
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

    # Guardar en knowledge_items
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
    db.refresh(knowledge)

    return {
        "id": str(knowledge.id),
        "question": data.question,
        "answer": a,
        "created_at": str(knowledge.created_at),
    }

@router.post("/learn", response_model=MemoryLearnResponse)
def learn(
    data: MemoryLearnRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    system_user = get_global_user(db)
    q = normalize_question(data.question)
    a = data.answer.strip()

    if not q:
        raise HTTPException(status_code=400, detail="La pregunta no puede estar vacía.")
    if not a:
        raise HTTPException(status_code=400, detail="La respuesta no puede estar vacía.")

    existing = db.query(QaMemory).filter(
        QaMemory.user_id == system_user.id,
        QaMemory.question == q,
        QaMemory.answer == a,
    ).first()

    if existing:
        existing.votes += 1
        db.commit()
        return MemoryLearnResponse(
            action="reinforced", question=q, answer=a,
            votes=existing.votes,
            message=f"Ya sabía eso. Reforcé la memoria (+1 voto → {existing.votes}).",
        )

    db.add(QaMemory(user_id=system_user.id, question=q, answer=a, votes=1))
    db.commit()
    return MemoryLearnResponse(
        action="created", question=q, answer=a,
        votes=1, message="¡Aprendido! Lo recordaré para todos.",
    )


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
            question_original=question, question_normalized=q,
            total_candidates=0, chosen_answer=None, candidates=[],
        )

    scores = {}
    candidates = []
    for r in rows:
        score = float(r.sim) * int(r.votes)
        scores[r.answer] = scores.get(r.answer, 0.0) + score
        candidates.append(MemoryCandidate(
            answer=r.answer, votes=r.votes,
            similarity=round(float(r.sim), 4),
            score=round(score, 4),
        ))

    candidates.sort(key=lambda x: x.score, reverse=True)
    top_sim = float(rows[0].sim)
    chosen = max(scores.items(), key=lambda x: x[1])[0] if top_sim >= 0.35 else None

    return MemoryStatsResponse(
        question_original=question, question_normalized=q,
        total_candidates=len(candidates),
        chosen_answer=chosen, candidates=candidates,
    )