import uuid
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.core.security import get_current_user
from app.models.user import User, KnowledgeItem, Embedding
from app.schemas.schemas import LearnRequest, KnowledgeOut
from app.services.embedding_service import get_embedding
from typing import List

router = APIRouter(tags=["Knowledge"])

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Divide texto en chunks con overlap para mejor RAG."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks if chunks else [text]


@router.post("/learn", response_model=KnowledgeOut)
async def learn(data: LearnRequest, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    # Crear knowledge item
    item = KnowledgeItem(
        created_by=current_user.id,
        title=data.title,
        content=data.content,
        source="manual",
        tags=data.tags or []
    )
    db.add(item)
    db.commit()
    db.refresh(item)

    # Generar embeddings por chunks
    chunks = chunk_text(data.content)
    try:
        for idx, chunk in enumerate(chunks):
            vector = await get_embedding(chunk)
            emb = Embedding(
                knowledge_id=item.id,
                chunk_index=idx,
                chunk_text=chunk,
                embedding=vector
            )
            db.add(emb)
        db.commit()
    except Exception as e:
        db.delete(item)
        db.commit()
        raise HTTPException(status_code=503, detail=f"Error generando embeddings: {str(e)}")

    result = KnowledgeOut.model_validate(item)
    result.creator_username = current_user.username
    return result


@router.get("/knowledge", response_model=List[KnowledgeOut])
def get_knowledge(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    items = db.query(KnowledgeItem).order_by(KnowledgeItem.created_at.desc()).all()
    result = []
    for item in items:
        out = KnowledgeOut.model_validate(item)
        if item.creator:
            out.creator_username = item.creator.username
        result.append(out)
    return result


@router.delete("/knowledge/{item_id}")
def delete_knowledge(item_id: uuid.UUID, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    item = db.query(KnowledgeItem).filter(KnowledgeItem.id == item_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="No encontrado")
    db.delete(item)
    db.commit()
    return {"ok": True}
