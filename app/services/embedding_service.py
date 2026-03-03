import httpx
from typing import List
from app.core.config import settings

async def get_embedding(text: str) -> List[float]:
    """
    Genera embedding usando nomic-embed-text via Ollama.
    Devuelve vector de 768 dimensiones.
    """
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{settings.OLLAMA_URL}/api/embeddings",
            json={"model": settings.EMBED_MODEL, "prompt": text}
        )
        response.raise_for_status()
        return response.json()["embedding"]


async def get_ollama_response(prompt: str, context: str) -> str:
    """
    Genera respuesta usando Mistral via Ollama con contexto RAG.
    """
    system_prompt = f"""Eres Nova, una IA asistente inteligente.
Responde SOLO basándote en el siguiente contexto proporcionado.
Si no tienes información suficiente, dilo claramente.
No inventes información.

CONTEXTO:
{context}
"""
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{settings.OLLAMA_URL}/api/generate",
            json={
                "model": settings.OLLAMA_MODEL,
                "prompt": prompt,
                "system": system_prompt,
                "stream": False
            }
        )
        response.raise_for_status()
        return response.json()["response"]
