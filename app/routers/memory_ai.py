import re
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.core.database import get_db
import re
import unicodedata
from app.core.security import get_current_user
from app.models.user import User

router = APIRouter(prefix="/memory", tags=["Memory AI"])

def normalize_question(q: str) -> str:
    q = q.strip().lower()

    # quitar tildes: quién -> quien
    q = unicodedata.normalize("NFKD", q)
    q = "".join(c for c in q if not unicodedata.combining(c))

    # quitar signos: ¿?¡! etc y dejar letras/números/espacios
    q = re.sub(r"[^a-z0-9\s]", " ", q)

    # quitar espacios
    q = re.sub(r"\s+", " ", q).strip()
    return q

class AskRequest(BaseModel):
    question: str

class TeachRequest(BaseModel):
    question: str
    answer: str

@router.post("/ask")
def ask(data: AskRequest, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    q = normalize_question(data.question)

    rows = db.execute(text("""
        SELECT question, answer, votes, similarity(question, :q) AS sim
        FROM qa_memory
        WHERE user_id = :uid
          AND question % :q
        ORDER BY sim DESC, votes DESC, updated_at DESC
        LIMIT 20
    """), {"q": q, "uid": str(current_user.id)}).fetchall()

    if not rows:
        return {"knows": False, "message": "No sé la respuesta. ¿Cuál es?"}

    # Agrupar por respuesta y escoger la mejor por score = sim * votes
    scores = {}
    for r in rows:
        score = float(r.sim) * int(r.votes)
        scores[r.answer] = scores.get(r.answer, 0.0) + score

    best_answer, best_score = max(scores.items(), key=lambda x: x[1])

    # Umbral: si la similitud es muy baja, mejor pedir confirmación
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
def teach(data: TeachRequest, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    q = normalize_question(data.question)
    a = data.answer.strip()

    db.execute(text("""
        INSERT INTO qa_memory (user_id, question, answer, votes, updated_at)
        VALUES (:uid, :q, :a, 1, NOW())
        ON CONFLICT (user_id, question, answer)
        DO UPDATE SET votes = qa_memory.votes + 1, updated_at = NOW()
    """), {"uid": str(current_user.id), "q": q, "a": a})

    db.commit()
    return {"message": "Aprendido con éxito. Gracias por enseñarme!"}