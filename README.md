# Nova Backend

FastAPI + PostgreSQL + pgvector + AWS

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
| GET  | /conversations | Historial de conversaciones |
| GET  | /conversations/{id}/messages | Mensajes de una conversación |
| POST | /memory/teach | Enseñar conocimiento nuevo |
| POST | /memory/ask | Consultar memoria de la IA |
| GET  | /memory/stats | Estadísticas de memoria |
| GET  | /knowledge | Listar todo el conocimiento |
| DELETE | /knowledge/{id} | Eliminar un item |
| POST | /upload-pdf | Subir PDF a la base de conocimiento |
| GET  | /pdfs | Listar PDFs subidos |
| POST | /polly | Texto a voz (Amazon Polly) |
| GET  | /pdf/ask | Consultar directamente los PDFs |

## Requisitos previos

- Python 3.12+
- RDS PostgreSQL con extensión pgvector
- Credenciales AWS (S3 + Polly)

## Sistema de IA

El sistema de inteligencia se basa en:

- **Memoria QA** — búsqueda por similitud con `pg_trgm` y sistema de votos
- **PDFs** — chunks de texto con búsqueda por keywords y stemming
- **Fallback** — si no hay información suficiente, Nova lo indica claramente