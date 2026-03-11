import re
import uuid
import boto3
import logging
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from PyPDF2 import PdfReader
from io import BytesIO

from app.core.database import get_db
from app.core.security import get_current_user
from app.core.config import settings
from app.models.user import User, PdfDocument, KnowledgeItem, PdfChunk
from app.schemas.schemas import PdfOut

router = APIRouter(tags=["Upload"])
logger = logging.getLogger(__name__)

def clean_pdf_text(text: str) -> str:
    text = re.sub(r'[~\-]\s*\d+\s*[~\-]', ' ', text)
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
    text = re.sub(r'\S+@\S+\.\S+', '', text)
    text = re.sub(r'http\S+', '', text)
    lines = [l.strip() for l in text.split('\n') if len(l.strip()) > 10]
    text = re.sub(r'[^\w\s.,;:áéíóúÁÉÍÓÚñÑüÜ¿?¡!()\-]', ' ', ' '.join(lines))
    return re.sub(r'\s+', ' ', text).strip()


def split_chunks(text: str, max_words: int = 180, overlap: int = 40):

    paragraphs = [
    p.strip()
    for p in text.split("\n")
        if len(p.strip()) > 10  # ← de 40 a 10
    ]

    chunks = []
    buffer = []

    for p in paragraphs:

        words = p.split()

        # párrafos muy largos
        if len(words) > max_words:

            i = 0
            while i < len(words):

                chunk = " ".join(words[i:i + max_words])

                if len(chunk.split()) > 20:
                    chunks.append(chunk)

                i += max_words - overlap

        else:

            buffer.extend(words)

            if len(buffer) >= max_words:

                chunks.append(" ".join(buffer))

                buffer = buffer[-overlap:]

    if buffer:
        chunks.append(" ".join(buffer))

    return chunks

@router.post("/upload-pdf", response_model=PdfOut)
async def upload_pdf(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
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
            region_name=settings.AWS_REGION,
        )
        s3.put_object(Bucket=settings.S3_BUCKET_NAME, Key=s3_key, Body=content)
    except Exception as e:
        logger.error(f"Error subiendo PDF a S3: {str(e)}")
        raise HTTPException(status_code=503, detail="Error subiendo el PDF. Intenta de nuevo más tarde.")

    # 2 — Extraer y limpiar texto
    try:
        reader = PdfReader(BytesIO(content))
        pages = len(reader.pages)
        raw_text = "\n".join([p.extract_text() or "" for p in reader.pages])
        full_text = clean_pdf_text(raw_text)
    except Exception:
        raise HTTPException(status_code=400, detail="No se pudo leer el PDF")

    if not full_text.strip():
        raise HTTPException(status_code=400, detail="El PDF no contiene texto extraíble.")

    # 3 — Guardar PdfDocument
    pdf_doc = PdfDocument(
        uploaded_by=current_user.id,
        filename=file.filename,
        s3_key=s3_key,
        file_size_bytes=len(content),
        page_count=pages,
        status="pending",
    )
    db.add(pdf_doc)
    db.commit()
    db.refresh(pdf_doc)

    try:
        knowledge_item = KnowledgeItem(
            created_by=current_user.id,
            source_pdf_id=pdf_doc.id,
            title=file.filename.replace(".pdf", ""),
            content=full_text[:1000],
            source="pdf",
            tags=[],
        )
        db.add(knowledge_item)
        db.commit()
        db.refresh(knowledge_item)

        chunks = split_chunks(full_text)
        if len(chunks) > 1200:
            chunks = chunks[:1200]

        for idx, chunk in enumerate(chunks):
            db.add(PdfChunk(
                pdf_id=pdf_doc.id,
                filename=file.filename,
                chunk_index=idx,
                chunk_text=chunk,
            ))

        db.commit()
        pdf_doc.status = "processed"
        pdf_doc.chunks_generated = len(chunks)
        db.commit()

    except Exception as e:
        logger.error(f"PDF processing failed: {str(e)}")
        pdf_doc.status = "error"
        db.commit()
        raise HTTPException(status_code=500, detail="Error procesando el PDF")

    db.refresh(pdf_doc)
    return PdfOut.model_validate(pdf_doc)


@router.get("/pdfs")
def get_pdfs(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return db.query(PdfDocument).filter(
        PdfDocument.uploaded_by == current_user.id
    ).order_by(PdfDocument.uploaded_at.desc()).all()