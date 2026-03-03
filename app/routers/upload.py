import uuid
import boto3
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from PyPDF2 import PdfReader
from io import BytesIO
from app.core.database import get_db
from app.core.security import get_current_user
from app.core.config import settings
from app.models.user import User, PdfDocument, KnowledgeItem, Embedding
from app.schemas.schemas import PdfOut
from app.services.embedding_service import get_embedding
from app.routers.knowledge import chunk_text

router = APIRouter(tags=["Upload"])

@router.post("/upload-pdf", response_model=PdfOut)
async def upload_pdf(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Solo se aceptan archivos PDF")

    content = await file.read()
    s3_key = f"pdfs/{current_user.id}/{uuid.uuid4()}_{file.filename}"

    # 1 — Subir a S3
    try:
        s3 = boto3.client(
            "s3",
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.AWS_REGION
        )
        s3.put_object(Bucket=settings.S3_BUCKET_NAME, Key=s3_key, Body=content)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Error subiendo a S3: {str(e)}")

    # 2 — Extraer texto del PDF
    try:
        reader = PdfReader(BytesIO(content))
        pages = len(reader.pages)
        full_text = "\n".join([p.extract_text() or "" for p in reader.pages])
    except Exception:
        raise HTTPException(status_code=400, detail="No se pudo leer el PDF")

    # 3 — Guardar registro en BD
    pdf_doc = PdfDocument(
        uploaded_by=current_user.id,
        filename=file.filename,
        s3_key=s3_key,
        file_size_bytes=len(content),
        page_count=pages,
        status="pending"
    )
    db.add(pdf_doc)
    db.commit()
    db.refresh(pdf_doc)

    # 4 — Crear knowledge items y embeddings por chunks
    chunks = chunk_text(full_text, chunk_size=400)
    total_chunks = 0

    try:
        knowledge_item = KnowledgeItem(
            created_by=current_user.id,
            source_pdf_id=pdf_doc.id,
            title=file.filename.replace(".pdf", ""),
            content=full_text[:1000],
            source="pdf",
            tags=[]
        )
        db.add(knowledge_item)
        db.commit()
        db.refresh(knowledge_item)

        for idx, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
            vector = await get_embedding(chunk)
            emb = Embedding(
                knowledge_id=knowledge_item.id,
                chunk_index=idx,
                chunk_text=chunk,
                embedding=vector
            )
            db.add(emb)
            total_chunks += 1

        db.commit()

        pdf_doc.status = "processed"
        pdf_doc.chunks_generated = total_chunks
        db.commit()

    except Exception as e:
        pdf_doc.status = "error"
        db.commit()
        raise HTTPException(status_code=500, detail=f"Error procesando PDF: {str(e)}")

    db.refresh(pdf_doc)
    return PdfOut.model_validate(pdf_doc)


@router.get("/pdfs")
def get_pdfs(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    return db.query(PdfDocument).filter(
        PdfDocument.uploaded_by == current_user.id
    ).order_by(PdfDocument.uploaded_at.desc()).all()
