import uuid
from datetime import datetime
from sqlalchemy import Column, String, Text, Boolean, Integer, BigInteger, ForeignKey, ARRAY, TIMESTAMP, Enum, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector
from app.core.database import Base
import enum

class UserRole(str, enum.Enum):
    admin = "admin"
    user  = "user"

class KnowledgeSource(str, enum.Enum):
    manual = "manual"
    pdf    = "pdf"
    chat   = "chat"

class PdfStatus(str, enum.Enum):
    pending   = "pending"
    processed = "processed"
    error     = "error"

class MessageRole(str, enum.Enum):
    user      = "user"
    assistant = "assistant"

class ConfidenceLevel(str, enum.Enum):
    high     = "high"
    medium   = "medium"
    inferred = "inferred"


class User(Base):
    __tablename__ = "users"
    id            = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username      = Column(String(50),  nullable=False, unique=True)
    email         = Column(String(100), nullable=False, unique=True)
    password_hash = Column(Text,        nullable=False)
    role          = Column(Enum(UserRole), nullable=False, default=UserRole.user)
    is_active     = Column(Boolean,     nullable=False, default=True)
    avatar_url    = Column(Text,        nullable=True)
    created_at    = Column(TIMESTAMP,   nullable=False, default=func.now())
    last_login    = Column(TIMESTAMP,   nullable=True)
    conversations  = relationship("Conversation",  back_populates="user",  cascade="all, delete-orphan")
    knowledge_items = relationship("KnowledgeItem", back_populates="creator")
    pdf_documents  = relationship("PdfDocument",   back_populates="uploader")


class Conversation(Base):
    __tablename__ = "conversations"
    id            = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id       = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    title         = Column(String(200), nullable=False, default="Nueva conversación")
    message_count = Column(Integer,     nullable=False, default=0)
    created_at    = Column(TIMESTAMP,   nullable=False, default=func.now())
    updated_at    = Column(TIMESTAMP,   nullable=False, default=func.now())
    user     = relationship("User",    back_populates="conversations")
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")


class Message(Base):
    __tablename__ = "messages"
    id               = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id  = Column(UUID(as_uuid=True), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False)
    role             = Column(Enum(MessageRole),     nullable=False)
    content          = Column(Text,                  nullable=False)
    confidence       = Column(Enum(ConfidenceLevel), nullable=True)
    tokens_used      = Column(Integer,               nullable=True)
    response_time_ms = Column(Integer,               nullable=True)
    created_at       = Column(TIMESTAMP,             nullable=False, default=func.now())
    conversation = relationship("Conversation", back_populates="messages")


class PdfDocument(Base):
    __tablename__ = "pdf_documents"
    id               = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    uploaded_by      = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    filename         = Column(String(255), nullable=False)
    s3_key           = Column(Text,        nullable=False, unique=True)
    file_size_bytes  = Column(BigInteger,  nullable=False)
    page_count       = Column(Integer,     nullable=False, default=0)
    status           = Column(Enum(PdfStatus), nullable=False, default=PdfStatus.pending)
    chunks_generated = Column(Integer,     nullable=True)
    uploaded_at      = Column(TIMESTAMP,   nullable=False, default=func.now())
    uploader        = relationship("User",          back_populates="pdf_documents")
    knowledge_items = relationship("KnowledgeItem", back_populates="source_pdf")


class KnowledgeItem(Base):
    __tablename__ = "knowledge_items"
    id            = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_by    = Column(UUID(as_uuid=True), ForeignKey("users.id",         ondelete="SET NULL"), nullable=True)
    source_pdf_id = Column(UUID(as_uuid=True), ForeignKey("pdf_documents.id", ondelete="SET NULL"), nullable=True)
    title         = Column(String(300),           nullable=False)
    content       = Column(Text,                  nullable=False)
    source        = Column(Enum(KnowledgeSource), nullable=False, default=KnowledgeSource.manual)
    tags          = Column(ARRAY(Text),           default=[])
    is_verified   = Column(Boolean,               nullable=False, default=False)
    use_count     = Column(Integer,               nullable=False, default=0)
    created_at    = Column(TIMESTAMP,             nullable=False, default=func.now())
    creator    = relationship("User",        back_populates="knowledge_items")
    source_pdf = relationship("PdfDocument", back_populates="knowledge_items")
    embeddings = relationship("Embedding",   back_populates="knowledge_item", cascade="all, delete-orphan")


class Embedding(Base):
    __tablename__ = "embeddings"
    id            = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    knowledge_id  = Column(UUID(as_uuid=True), ForeignKey("knowledge_items.id", ondelete="CASCADE"), nullable=False)
    chunk_index   = Column(Integer,     nullable=False, default=0)
    chunk_text    = Column(Text,        nullable=False)
    embedding     = Column(Vector(768), nullable=False)
    model_version = Column(String(50),  nullable=True, default="nomic-embed-text")
    created_at    = Column(TIMESTAMP,   nullable=False, default=func.now())
    knowledge_item = relationship("KnowledgeItem", back_populates="embeddings")
