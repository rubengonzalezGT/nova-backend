from pydantic import BaseModel, EmailStr
from typing import Optional, List
from datetime import datetime
from uuid import UUID

# ── Auth ─────────────────────────────────────────────────────
class RegisterRequest(BaseModel):
    username: str
    email: EmailStr
    password: str

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class UserOut(BaseModel):
    id: UUID
    username: str
    email: str
    role: str
    is_active: bool
    avatar_url: Optional[str]
    created_at: datetime
    class Config: from_attributes = True

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut

# ── Chat ─────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[UUID] = None

class MessageOut(BaseModel):
    id: UUID
    conversation_id: UUID
    role: str
    content: str
    confidence: Optional[str]
    tokens_used: Optional[int]
    response_time_ms: Optional[int]
    created_at: datetime
    class Config: from_attributes = True

class ChatResponse(BaseModel):
    message: MessageOut
    conversation_id: UUID

# ── Knowledge ────────────────────────────────────────────────
class LearnRequest(BaseModel):
    title: str
    content: str
    tags: Optional[List[str]] = []

class KnowledgeOut(BaseModel):
    id: UUID
    title: str
    content: str
    source: str
    tags: List[str]
    is_verified: bool
    use_count: int
    created_at: datetime
    creator_username: Optional[str] = None
    class Config: from_attributes = True

# ── Upload ───────────────────────────────────────────────────
class PdfOut(BaseModel):
    id: UUID
    filename: str
    status: str
    page_count: int
    chunks_generated: Optional[int]
    uploaded_at: datetime
    class Config: from_attributes = True

# ── Polly ────────────────────────────────────────────────────
class PollyRequest(BaseModel):
    text: str
