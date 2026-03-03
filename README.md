# Nova Backend 🧠

FastAPI + PostgreSQL + pgvector + Ollama

## Setup

```bash
# 1. Crear entorno virtual
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Configurar variables de entorno
cp .env.example .env
# Editar .env con tus credenciales reales

# 4. Correr el servidor
uvicorn app.main:app --reload
```

Abre: http://localhost:8000/docs

## Endpoints

| Método | Ruta | Descripción |
|--------|------|-------------|
| POST | /auth/register | Registrar usuario |
| POST | /auth/login | Login → JWT |
| POST | /chat | Enviar mensaje a la IA |
| POST | /learn | Enseñar conocimiento nuevo |
| GET  | /knowledge | Listar todo el conocimiento |
| POST | /upload-pdf | Subir PDF → RAG automático |
| POST | /polly | Texto a voz (Amazon Polly) |

## Requisitos previos

- Python 3.12+
- Ollama corriendo con `mistral` y `nomic-embed-text`
- RDS PostgreSQL con pgvector
- Credenciales AWS (S3 + Polly)
