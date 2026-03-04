import re
import unicodedata
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.core.database import get_db
from app.core.security import get_current_user
from app.models.user import User
from app.schemas.schemas import (
    MemoryLearnRequest,
    MemoryLearnResponse,
    MemoryCandidate,
    MemoryStatsResponse,
)

router = APIRouter(prefix="/memory", tags=["Memory AI"])

SYSTEM_EMAIL = "system@nova.com"  # tu usuario global

def get_global_user_id(db: Session) -> str:
    row = db.execute(
        text("SELECT id FROM users WHERE email = :email LIMIT 1"),
        {"email": SYSTEM_EMAIL},
    ).first()
    if not row:
        raise HTTPException(status_code=500, detail=f"No existe usuario global: {SYSTEM_EMAIL}")
    return str(row.id)

def normalize_question(q: str) -> str:
    q = q.strip().lower()

    # quitar tildes: quién -> quien
    q = unicodedata.normalize("NFKD", q)
    q = "".join(c for c in q if not unicodedata.combining(c))

    # quitar signos: ¿?¡! etc y dejar letras/números/espacios
    q = re.sub(r"[^a-z0-9\s]", " ", q)

    # normalizar espacios
    q = re.sub(r"\s+", " ", q).strip()
    return q

class AskRequest(BaseModel):
    question: str

class TeachRequest(BaseModel):
    question: str
    answer: str

@router.post("/ask")
def ask(
    data: AskRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)  # solo para proteger endpoint
):
    uid = get_global_user_id(db)  # 👈 memoria global
    q = normalize_question(data.question)

    rows = db.execute(text("""
        SELECT question, answer, votes, similarity(question, :q) AS sim
        FROM qa_memory
        WHERE user_id = :uid
          AND question % :q
        ORDER BY sim DESC, votes DESC, updated_at DESC
        LIMIT 20
    """), {"q": q, "uid": uid}).fetchall()

    if not rows:
        return {
            "knows": False,
            "message": "No sé la respuesta a tu pregunta. ¿Me puedes ayudar diciendo lo que entiendes sobre tu pregunta?"
        }

    # Agrupar por respuesta y escoger la mejor por score = sim * votes
    scores = {}
    for r in rows:
        score = float(r.sim) * int(r.votes)
        scores[r.answer] = scores.get(r.answer, 0.0) + score

    best_answer, _ = max(scores.items(), key=lambda x: x[1])

    top_sim = float(rows[0].sim)
    if top_sim < 0.35:
        return {
            "knows": False,
            "message": "No estoy seguro (pregunta muy distinta). ¿Cuál es la respuesta correcta?"
        }

    return {
        "knows": True,
        "answer": best_answer,
        "message": f"Respuesta aprendida (similitud top={top_sim:.2f})."
    }

@router.post("/teach")
def teach(
    data: TeachRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)  # solo para proteger endpoint
):
    uid = get_global_user_id(db)  # 👈 memoria global
    q = normalize_question(data.question)
    a = data.answer.strip()

    if not a:
        raise HTTPException(status_code=422, detail="La respuesta no puede ir vacía")

    db.execute(text("""
        INSERT INTO qa_memory (user_id, question, answer, votes, updated_at)
        VALUES (:uid, :q, :a, 1, NOW())
        ON CONFLICT (user_id, question, answer)
        DO UPDATE SET votes = qa_memory.votes + 1,
                      updated_at = NOW()
    """), {"uid": uid, "q": q, "a": a})

    db.commit()
    return {"message": "Aprendido con éxito. Gracias por enseñarme!"}


@router.post("/learn", response_model=MemoryLearnResponse)
def learn(
    data: MemoryLearnRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    system_id = get_global_user_id(db)
    norm_q    = normalize_question(data.question)
    answer    = data.answer.strip()

    if not norm_q:
        raise HTTPException(status_code=400, detail="La pregunta no puede estar vacía.")
    if not answer:
        raise HTTPException(status_code=400, detail="La respuesta no puede estar vacía.")

    existing = db.execute(
        text("""
            SELECT id, votes
            FROM qa_memory
            WHERE user_id = :uid
              AND question = :q
              AND answer   = :a
            LIMIT 1
        """),
        {"uid": system_id, "q": norm_q, "a": answer},
    ).first()

    if existing:
        db.execute(
            text("""
                UPDATE qa_memory
                SET votes      = votes + 1,
                    updated_at = NOW()
                WHERE id = :id
            """),
            {"id": existing.id},
        )
        db.commit()
        return MemoryLearnResponse(
            action   = "reinforced",
            question = norm_q,
            answer   = answer,
            votes    = existing.votes + 1,
            message  = f"Ya sabía eso. Reforcé la memoria (+1 voto → {existing.votes + 1}).",
        )

    db.execute(
        text("""
            INSERT INTO qa_memory (user_id, question, answer, votes)
            VALUES (:uid, :q, :a, 1)
        """),
        {"uid": system_id, "q": norm_q, "a": answer},
    )
    db.commit()
    return MemoryLearnResponse(
        action   = "created",
        question = norm_q,
        answer   = answer,
        votes    = 1,
        message  = "¡Aprendido! Lo recordaré para todos.",
    )

@router.get("/stats", response_model=MemoryStatsResponse)
def stats(
    question: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    uid   = get_global_user_id(db)
    norm_q = normalize_question(question)

    rows = db.execute(text("""
        SELECT answer, votes, similarity(question, :q) AS sim
        FROM qa_memory
        WHERE user_id = :uid
          AND question % :q
        ORDER BY sim DESC, votes DESC
        LIMIT 20
    """), {"q": norm_q, "uid": uid}).fetchall()

    if not rows:
        return MemoryStatsResponse(
            question_original   = question,
            question_normalized = norm_q,
            total_candidates    = 0,
            chosen_answer       = None,
            candidates          = [],
        )

    # Calcular score = sim * votes (igual que /ask)
    scores = {}
    candidates = []
    for r in rows:
        score = float(r.sim) * int(r.votes)
        scores[r.answer] = scores.get(r.answer, 0.0) + score
        candidates.append(MemoryCandidate(
            answer     = r.answer,
            votes      = r.votes,
            similarity = round(float(r.sim), 4),
            score      = round(score, 4),
        ))

    # Ordenar candidatos por score descendente
    candidates.sort(key=lambda x: x.score, reverse=True)

    top_sim = float(rows[0].sim)
    chosen  = max(scores.items(), key=lambda x: x[1])[0] if top_sim >= 0.35 else None

    return MemoryStatsResponse(
        question_original   = question,
        question_normalized = norm_q,
        total_candidates    = len(candidates),
        chosen_answer       = chosen,
        candidates          = candidates,
    )
