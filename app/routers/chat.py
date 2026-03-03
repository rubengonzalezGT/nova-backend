import time
import uuid
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.core.database import get_db
from app.core.security import get_current_user
from app.models.user import User, Conversation, Message, KnowledgeItem, Embedding, ConfidenceLevel
from app.schemas.schemas import ChatRequest, ChatResponse, MessageOut
from app.services.embedding_service import get_embedding, get_ollama_response

router = APIRouter(tags=["Chat"])

@router.post("/chat", response_model=ChatResponse)
async def chat(data: ChatRequest, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    start = time.time()

    # 1 — Obtener o crear conversación
    if data.conversation_id:
        conversation = db.query(Conversation).filter(
            Conversation.id == data.conversation_id,
            Conversation.user_id == current_user.id
        ).first()
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversación no encontrada")
    else:
        conversation = Conversation(
            user_id=current_user.id,
            title=data.message[:50]
        )
        db.add(conversation)
        db.commit()
        db.refresh(conversation)

    # 2 — Guardar mensaje del usuario
    user_msg = Message(
        conversation_id=conversation.id,
        role="user",
        content=data.message
    )
    db.add(user_msg)
    db.commit()

    # 3 — Generar embedding de la pregunta
    try:
        query_vector = await get_embedding(data.message)
    except Exception:
        raise HTTPException(status_code=503, detail="Ollama no disponible. Verifica que esté corriendo.")

    # 4 — Búsqueda semántica en pgvector
    vector_str = "[" + ",".join(map(str, query_vector)) + "]"
    results = db.execute(text("""
        SELECT e.chunk_text, k.title,
               1 - (e.embedding <=> :vec::vector) AS similarity
        FROM embeddings e
        JOIN knowledge_items k ON e.knowledge_id = k.id
        WHERE 1 - (e.embedding <=> :vec::vector) > 0.45
        ORDER BY similarity DESC
        LIMIT 5
    """), {"vec": vector_str}).fetchall()

    # 5 — Determinar confianza
    if not results:
        confidence = ConfidenceLevel.inferred
        context = "No tengo información específica sobre este tema."
    elif results[0].similarity > 0.85:
        confidence = ConfidenceLevel.high
        context = "\n\n".join([f"[{r.title}]: {r.chunk_text}" for r in results])
    elif results[0].similarity > 0.65:
        confidence = ConfidenceLevel.medium
        context = "\n\n".join([f"[{r.title}]: {r.chunk_text}" for r in results])
    else:
        confidence = ConfidenceLevel.inferred
        context = "\n\n".join([f"[{r.title}]: {r.chunk_text}" for r in results])

    # Actualizar use_count de los knowledge items usados
    if results:
        for r in results:
            db.execute(text("""
                UPDATE knowledge_items k
                SET use_count = use_count + 1
                FROM embeddings e
                WHERE e.knowledge_id = k.id AND e.chunk_text = :chunk
            """), {"chunk": r.chunk_text})
        db.commit()

    # 6 — Generar respuesta con Mistral
    try:
        response_text = await get_ollama_response(data.message, context)
    except Exception:
        response_text = "Lo siento, hubo un error al generar la respuesta."
        confidence = ConfidenceLevel.inferred

    elapsed_ms = int((time.time() - start) * 1000)

    # 7 — Guardar respuesta de la IA
    ai_msg = Message(
        conversation_id=conversation.id,
        role="assistant",
        content=response_text,
        confidence=confidence,
        response_time_ms=elapsed_ms
    )
    db.add(ai_msg)
    db.commit()
    db.refresh(ai_msg)

    return ChatResponse(
        message=MessageOut.model_validate(ai_msg),
        conversation_id=conversation.id
    )


@router.get("/conversations")
def get_conversations(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    return db.query(Conversation).filter(
        Conversation.user_id == current_user.id
    ).order_by(Conversation.updated_at.desc()).all()


@router.get("/conversations/{conversation_id}/messages")
def get_messages(conversation_id: uuid.UUID, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    conversation = db.query(Conversation).filter(
        Conversation.id == conversation_id,
        Conversation.user_id == current_user.id
    ).first()
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversación no encontrada")
    return db.query(Message).filter(
        Message.conversation_id == conversation_id
    ).order_by(Message.created_at.asc()).all()
