import time
import uuid
import re
import unicodedata
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.core.security import get_current_user
from app.models.user import User, Conversation, Message, QaMemory, ConfidenceLevel
from app.schemas.schemas import ChatRequest, ChatResponse, MessageOut
from sqlalchemy import text

router = APIRouter(tags=["Chat"])

SYSTEM_EMAIL = "system@nova.com"


# ── Helpers ───────────────────────────────────────────────────

def normalize_question(q: str) -> str:
    q = q.strip().lower()
    q = unicodedata.normalize("NFKD", q)
    q = "".join(c for c in q if not unicodedata.combining(c))
    q = re.sub(r"[^a-z0-9\s]", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q


def get_global_user(db: Session) -> User:
    user = db.query(User).filter(User.email == SYSTEM_EMAIL).first()
    if not user:
        raise HTTPException(status_code=500, detail=f"No existe usuario global: {SYSTEM_EMAIL}")
    return user


def query_memory(db: Session, question: str) -> dict:
    system_user = get_global_user(db)
    q = normalize_question(question)

    rows = db.execute(text("""
        SELECT question, answer, votes, similarity(question, :q) AS sim
        FROM qa_memory
        WHERE user_id = :uid AND question % :q
        ORDER BY sim DESC, votes DESC, updated_at DESC
        LIMIT 20
    """), {"q": q, "uid": str(system_user.id)}).fetchall()

    if not rows:
        return {"knows": False, "answer": None, "confidence": "inferred", "similarity": 0.0}

    scores = {}
    for r in rows:
        scores[r.answer] = scores.get(r.answer, 0.0) + float(r.sim) * max(int(r.votes), 1)

    best_answer, _ = max(scores.items(), key=lambda x: x[1])
    top_sim = float(rows[0].sim)

    if top_sim < 0.35:
        return {"knows": False, "answer": None, "confidence": "inferred", "similarity": top_sim}

    confidence = "high" if top_sim >= 0.85 else "medium" if top_sim >= 0.60 else "inferred"
    return {"knows": True, "answer": best_answer, "confidence": confidence, "similarity": top_sim}


# ── Endpoints ─────────────────────────────────────────────────

@router.post("/chat", response_model=ChatResponse)
def chat(
    data: ChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    start = time.time()

    # 1 — Obtener o crear conversación
    if data.conversation_id:
        conversation = db.query(Conversation).filter(
            Conversation.id == data.conversation_id,
            Conversation.user_id == current_user.id,
        ).first()
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversación no encontrada")
    else:
        conversation = Conversation(user_id=current_user.id, title=data.message[:50])
        db.add(conversation)
        db.commit()
        db.refresh(conversation)

    # 2 — Guardar mensaje del usuario
    db.add(Message(conversation_id=conversation.id, role="user", content=data.message))
    db.commit()

    # 3 — Consultar memoria
    result = query_memory(db, data.message)

    # 4 — Construir respuesta
    response_text = result["answer"] if result["knows"] else \
        "No tengo información sobre eso todavía. ¿Me puedes enseñar? Usa el panel de conocimiento."

    confidence_map = {
        "high": ConfidenceLevel.high,
        "medium": ConfidenceLevel.medium,
        "inferred": ConfidenceLevel.inferred,
    }
    elapsed_ms = int((time.time() - start) * 1000)

    # 5 — Guardar respuesta
    ai_msg = Message(
        conversation_id=conversation.id,
        role="assistant",
        content=response_text,
        confidence=confidence_map[result["confidence"]],
        response_time_ms=elapsed_ms,
    )
    db.add(ai_msg)
    db.commit()
    db.refresh(ai_msg)

    return ChatResponse(message=MessageOut.model_validate(ai_msg), conversation_id=conversation.id)


@router.get("/conversations")
def get_conversations(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return db.query(Conversation).filter(
        Conversation.user_id == current_user.id
    ).order_by(Conversation.updated_at.desc()).all()


@router.get("/conversations/{conversation_id}/messages")
def get_messages(
    conversation_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    conversation = db.query(Conversation).filter(
        Conversation.id == conversation_id,
        Conversation.user_id == current_user.id,
    ).first()
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversación no encontrada")
    return db.query(Message).filter(
        Message.conversation_id == conversation_id
    ).order_by(Message.created_at.asc()).all()


@router.get("/historial")
def get_historial(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    conversaciones = db.query(Conversation).filter(
        Conversation.user_id == current_user.id
    ).order_by(Conversation.updated_at.desc()).all()

    return [
        {
            "conversation_id": str(conv.id),
            "title": conv.title,
            "message_count": conv.message_count,
            "created_at": str(conv.created_at),
            "updated_at": str(conv.updated_at),
            "messages": [
                {
                    "role": m.role,
                    "content": m.content,
                    "confidence": m.confidence,
                    "created_at": str(m.created_at),
                }
                for m in db.query(Message).filter(
                    Message.conversation_id == conv.id
                ).order_by(Message.created_at.asc()).all()
            ],
        }
        for conv in conversaciones
    ]