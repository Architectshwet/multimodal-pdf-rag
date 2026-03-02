# Multimodal PDF RAG (FastAPI + OpenAI + Qdrant + PostgreSQL)

This project implements a multimodal PDF RAG pipeline with two stages: indexing and question answering.
During indexing, each PDF page is converted to an image, described by an OpenAI vision model, embedded, and stored in Qdrant with page metadata. During chat, the user query is rewritten from conversation history, top-k pages are retrieved, retrieved page images are sent to the answer model, and the final response is streamed.

## Features

- **Indexing flow**:
  `PDF page -> page image -> vision description -> embedding -> Qdrant upsert`.
- **Page-level multimodal representation**:
  each vectorized unit is one PDF page with an LLM-generated full-page description and metadata (`source`, `page_number`, `image_path`).
- **History-aware query rewrite**:
  chat history + user message are transformed into a retrieval-ready rewritten query before search.
- **Top-k vector retrieval from Qdrant**:
  rewritten query embedding is used to fetch the most relevant pages for grounding.
- **Image-grounded answer generation**:
  answer generation runs on retrieved page images plus user query and rewritten query context.
- **Streaming response protocol**:
  FastAPI SSE stream emits structured events (`session`, `retrieval`, `token`, `end_of_response`, `error`).
- **Deterministic re-indexing**:
  `POST /index` clears previous vectors and rendered page images before rebuilding the index.
- **Index inspection endpoints**:
  page-level debugging via `/index/stats` and `/index/pages/{page_number}` for `page_content` and metadata validation.
- **Remote chat memory**:
  async PostgreSQL persistence with connection pooling for thread-scoped multi-turn chat state.
- **Built-in Web UI**:
  chat interface is served from `GET /web`.

## Architecture

1. **API and lifecycle layer** (`multimodal/server.py`):
   initializes core services in lifespan, validates pooled Postgres connectivity, and serves index/chat/debug routes.
2. **Indexing layer** (`multimodal/services/indexing_service.py`):
   loads the single PDF from `uploads/`, creates page images, generates page descriptions, and prepares vector documents.
3. **Model layer** (`multimodal/services/openai_service.py`):
   runs page description generation, history-aware query rewrite, and image-grounded answer generation.
4. **Retrieval layer** (`multimodal/services/qdrant_service.py`):
   manages collection reset, embedding upsert, top-k similarity retrieval, and page inspection reads.
5. **Memory layer** (`multimodal/db/*`, `multimodal/services/postgres_db_service.py`):
   stores and fetches conversation history from remote PostgreSQL using async pooled connections.

## Use Cases

- Multimodal PDF Q&A where answers must be grounded on page images (not only extracted text).
- Validation of LLM page descriptions using page-wise inspection (`/index/pages/{page_number}`).
- Multi-turn assistant workflows that require query rewriting before retrieval.
- Reference implementation for OpenAI multimodal RAG with FastAPI streaming, Qdrant retrieval, and remote PostgreSQL memory.

## Run (Docker)

1. Create environment file:

```bash
cp .env.example .env
```

2. Set required variables in `.env`:
   `OPENAI_API_KEY`, `POSTGRESQL_URL`

3. Put one PDF into `uploads/` (for example `uploads/extracted.pdf`).

4. Start dev stack:

```bash
docker-compose -f docker-compose.dev.yml up --build
```

5. Build index:

```bash
curl -X POST "http://localhost:8000/index"
```

6. Use the app:
   open `http://localhost:8000/web` for Chat UI.
   check `http://localhost:8000/health` for health status.
   API docs are available at `http://localhost:8000/docs`.

## Docker Lifecycle Commands

1. Stop containers:

```bash
docker-compose -f docker-compose.dev.yml down
```

2. Start again (without rebuild):

```bash
docker-compose -f docker-compose.dev.yml up
```

3. Rebuild and start:

```bash
docker-compose -f docker-compose.dev.yml up --build
```

4. Optional full cleanup:

```bash
docker-compose -f docker-compose.dev.yml down -v
```

## Debug Endpoints

1. Index stats:

```bash
curl "http://localhost:8000/index/stats"
```

2. Inspect one indexed page:
   returns `status`, `page_number`, `page_content` (embedded text), `metadata`, and `token_usage`.

```bash
curl "http://localhost:8000/index/pages/1"
```
