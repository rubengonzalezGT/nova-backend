from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import auth, chat, knowledge, upload, polly, pdf_ask
from app.routers import memory_ai

app = FastAPI(
    title="Nova AI — Backend",
    description="API para Nova: IA conversacional con RAG local",
    version="1.0.0"
)

# ── CORS ─────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:4200",
        "https://nova-front.onrender.com",
        "https://nova-backend-alos.onrender.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ──────────────────────────────────────────────────
app.include_router(auth.router)
app.include_router(chat.router)
app.include_router(knowledge.router)
app.include_router(upload.router)
app.include_router(polly.router)
app.include_router(memory_ai.router)
app.include_router(pdf_ask.router)

@app.get("/")
def root():
    return {"status": "Nova API corriendo", "version": "1.0.0"}

@app.get("/health")
def health():
    return {"status": "ok"}