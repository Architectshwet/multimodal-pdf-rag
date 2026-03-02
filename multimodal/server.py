import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from multimodal.config import config
from multimodal.db.db_singleton import DatabaseSingleton
from multimodal.services.chat_service import ChatService
from multimodal.services.indexing_service import IndexingService
from multimodal.services.openai_service import OpenAIService
from multimodal.services.postgres_db_service import PostgresDBService
from multimodal.services.qdrant_service import QdrantService, get_qdrant_service
from multimodal.utils.logger import get_logger

logger = get_logger(__name__)


WEB_UI = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Multimodal PDF RAG Chatbot</title>
  <style>
    :root {
      --bg-1: #081a29;
      --bg-2: #0f2f43;
      --panel: #f8fafc;
      --line: #d4dce4;
      --text: #0f172a;
      --muted: #475569;
      --brand: #0f766e;
      --user: #dbeafe;
      --assistant: #ffffff;
      --system: #ecfeff;
    }
    * {
      box-sizing: border-box;
    }
    body {
      margin: 0;
      font-family: "DM Sans", "Segoe UI", sans-serif;
      background:
        radial-gradient(900px 500px at 10% 0%, #124d6a 0%, transparent 60%),
        linear-gradient(160deg, var(--bg-1), var(--bg-2));
      min-height: 100vh;
      color: var(--text);
      padding: 16px;
    }
    .shell {
      max-width: 1100px;
      margin: 0 auto;
      background: var(--panel);
      border-radius: 18px;
      overflow: hidden;
      box-shadow: 0 18px 50px rgba(0, 0, 0, 0.22);
    }
    .header {
      padding: 18px 20px 16px;
      background: linear-gradient(130deg, #0f172a, #115e59 60%, #0f766e);
      color: #fff;
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 14px;
    }
    .title h1 {
      margin: 0;
      font-size: 30px;
      line-height: 1.05;
      letter-spacing: 0.2px;
    }
    .title p {
      margin: 8px 0 0;
      font-size: 14px;
      opacity: 0.92;
    }
    .thread-wrap {
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }
    .thread-pill {
      border: 1px solid rgba(255, 255, 255, 0.4);
      border-radius: 999px;
      padding: 7px 12px;
      font-size: 12px;
      background: rgba(255, 255, 255, 0.12);
      max-width: 360px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .new-thread-btn {
      border: 1px solid rgba(255, 255, 255, 0.45);
      background: rgba(255, 255, 255, 0.16);
      color: #fff;
      border-radius: 999px;
      padding: 7px 12px;
      font-size: 12px;
      font-weight: 600;
      cursor: pointer;
    }
    .new-thread-btn:hover {
      background: rgba(255, 255, 255, 0.24);
    }
    .content {
      padding: 14px;
      display: grid;
      gap: 10px;
    }
    .status-note {
      border: 1px solid var(--line);
      border-radius: 10px;
      background: #ffffff;
      padding: 10px 12px;
      font-size: 13px;
      color: var(--muted);
    }
    #messages {
      border: 1px solid var(--line);
      border-radius: 12px;
      background: #edf2f7;
      height: 62vh;
      overflow: auto;
      padding: 12px;
    }
    .msg {
      margin: 10px 0;
    }
    .msg.user {
      text-align: right;
    }
    .bubble {
      display: inline-block;
      max-width: 84%;
      padding: 10px 12px;
      border-radius: 12px;
      border: 1px solid #cbd5e1;
      background: var(--assistant);
      white-space: pre-wrap;
      text-align: left;
      box-shadow: 0 1px 1px rgba(15, 23, 42, 0.06);
    }
    .msg.user .bubble {
      background: var(--user);
      border-color: #93c5fd;
    }
    .msg.system .bubble {
      background: var(--system);
      border-color: #99f6e4;
      color: #0f172a;
      font-size: 12px;
    }
    .row {
      display: flex;
      gap: 8px;
      align-items: center;
    }
    .row input {
      flex: 1;
      border: 1px solid #94a3b8;
      border-radius: 10px;
      padding: 12px;
      font-size: 14px;
      outline: none;
    }
    .row input:focus {
      border-color: #0ea5e9;
      box-shadow: 0 0 0 3px rgba(14, 165, 233, 0.15);
    }
    .row button {
      width: 130px;
      border: 0;
      border-radius: 10px;
      background: var(--brand);
      color: #fff;
      font-weight: 700;
      cursor: pointer;
      height: 44px;
    }
    .row button:disabled {
      opacity: 0.65;
      cursor: default;
    }
    .hint {
      color: var(--muted);
      font-size: 12px;
      margin-top: -4px;
    }
    @media (max-width: 840px) {
      .header {
        flex-direction: column;
        align-items: flex-start;
      }
      .thread-wrap {
        width: 100%;
        justify-content: flex-start;
      }
      .title h1 {
        font-size: 24px;
      }
      #messages {
        height: 56vh;
      }
      .row button {
        width: 100px;
      }
    }
  </style>
</head>
<body>
  <div class="shell">
    <div class="header">
      <div class="title">
        <h1>Multimodal PDF RAG Chatbot</h1>
        <p>Chat with your indexed PDF using retrieval-grounded multimodal responses.</p>
      </div>
      <div class="thread-wrap">
        <div class="thread-pill" id="threadPill">Thread: (new)</div>
        <button class="new-thread-btn" id="newThreadBtn" type="button">New Thread</button>
      </div>
    </div>
    <div class="content">
      <div class="status-note">
        Indexing is managed via API (`POST /index`). This web screen is chat-only.
      </div>
      <div id="messages"></div>
      <div class="row">
        <input id="q" placeholder="Ask a question about the indexed PDF..." />
        <button id="sendBtn">Send</button>
      </div>
      <div class="hint">Press Enter to send. If no data is indexed, the API will ask you to run index first.</div>
    </div>
  </div>

<script>
let threadId = null;
const messages = document.getElementById("messages");
const threadPill = document.getElementById("threadPill");
const newThreadBtn = document.getElementById("newThreadBtn");
const qInput = document.getElementById("q");
const sendBtn = document.getElementById("sendBtn");
let isSending = false;
const THREAD_STORAGE_KEY = "multimodal_pdf_rag_thread_id";

function generateThreadId(length = 10) {
  const alphabet = "abcdefghijklmnopqrstuvwxyz0123456789";
  let out = "";
  for (let i = 0; i < length; i++) {
    out += alphabet[Math.floor(Math.random() * alphabet.length)];
  }
  return out;
}

function isValidThreadId(id) {
  return typeof id === "string" && /^[a-z0-9]{10}$/.test(id);
}

function ensureThreadId() {
  let saved = null;
  try {
    saved = localStorage.getItem(THREAD_STORAGE_KEY);
  } catch (err) {
    saved = null;
  }

  if (!isValidThreadId(saved)) {
    saved = generateThreadId(10);
    try {
      localStorage.setItem(THREAD_STORAGE_KEY, saved);
    } catch (err) {
      // ignore storage failures
    }
  }

  setThread(saved);
}

function setThread(id) {
  threadId = id || null;
  if (threadId) {
    try {
      localStorage.setItem(THREAD_STORAGE_KEY, threadId);
    } catch (err) {
      // ignore storage failures
    }
  }
  threadPill.textContent = "Thread: " + (threadId || "(new)");
}

function addMessage(role, text) {
  const box = document.createElement("div");
  box.className = "msg " + role;
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;
  box.appendChild(bubble);
  messages.appendChild(box);
  messages.scrollTop = messages.scrollHeight;
  return bubble;
}

async function sendMessage() {
  if (isSending) return;
  const message = (qInput.value || "").trim();
  if (!message) return;

  isSending = true;
  sendBtn.disabled = true;
  qInput.disabled = true;
  qInput.value = "";
  addMessage("user", message);
  const aiBubble = addMessage("assistant", "");

  const payload = {
    message: message,
    thread_id: threadId
  };

  try {
    const res = await fetch("/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    if (!res.ok || !res.body) {
      const text = await res.text();
      aiBubble.textContent = "Error: " + (text || "request failed");
      return;
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const eventBlocks = buffer.split("\\n\\n");
      buffer = eventBlocks.pop() || "";

      for (const block of eventBlocks) {
        const dataLines = block
          .split("\\n")
          .filter((line) => line.startsWith("data: "))
          .map((line) => line.slice(6).trim());
        if (!dataLines.length) continue;

        const raw = dataLines.join("\\n");
        let evt = null;
        try {
          evt = JSON.parse(raw);
        } catch (parseErr) {
          aiBubble.textContent = "Error: failed to parse stream event";
          continue;
        }

        if (evt.type === "session") {
          setThread(evt.thread_id);
        } else if (evt.type === "retrieval") {
          const sourceCount = Array.isArray(evt.sources) ? evt.sources.length : 0;
          const rewritten = evt.rewritten_query || "";
          addMessage("system", "Retrieved " + sourceCount + " pages. Rewritten query: " + rewritten);
        } else if (evt.type === "token") {
          aiBubble.textContent += evt.content || "";
          messages.scrollTop = messages.scrollHeight;
        } else if (evt.type === "end_of_response") {
          if (!aiBubble.textContent.trim()) {
            aiBubble.textContent = evt.content || "";
          }
        } else if (evt.type === "error") {
          aiBubble.textContent = "Error: " + (evt.error || "unknown");
        }
      }
    }
  } catch (err) {
    aiBubble.textContent = "Error: " + String(err);
  } finally {
    isSending = false;
    sendBtn.disabled = false;
    qInput.disabled = false;
    qInput.focus();
  }
}

sendBtn.addEventListener("click", sendMessage);
qInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    e.preventDefault();
    sendMessage();
  }
});
newThreadBtn.addEventListener("click", () => {
  const nextThreadId = generateThreadId(10);
  setThread(nextThreadId);
  messages.innerHTML = "";
  addMessage("system", "Started new thread: " + nextThreadId);
});

ensureThreadId();
</script>
</body>
</html>
"""


db_service = PostgresDBService()
openai_service: OpenAIService | None = None
qdrant_service: QdrantService | None = None


class ChatRequest(BaseModel):
    message: str
    thread_id: str | None = None


def get_openai() -> OpenAIService:
    if openai_service is None:
        raise RuntimeError("OpenAI service not initialized")
    return openai_service


def create_indexing_service() -> IndexingService:
    return IndexingService(
        qdrant_service=get_qdrant(),
        openai_service=get_openai(),
    )


def create_chat_service() -> ChatService:
    return ChatService(
        db_service=db_service,
        qdrant_service=get_qdrant(),
        openai_service=get_openai(),
    )


def get_qdrant() -> QdrantService:
    if qdrant_service is None:
        raise RuntimeError("Qdrant service not initialized")
    return qdrant_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    global openai_service, qdrant_service

    logger.info("Starting Multimodal PDF RAG API")

    config.page_images_dir.mkdir(parents=True, exist_ok=True)
    config.qdrant_path.mkdir(parents=True, exist_ok=True)

    await db_service.initialize()
    pool = await DatabaseSingleton.get_pool()
    async with pool.acquire() as conn:
        await conn.fetchval("SELECT 1")
    logger.info("PostgreSQL connection pool initialized and connectivity check passed")

    openai_service = OpenAIService()
    qdrant_service = get_qdrant_service()
    logger.info("Core services initialized (OpenAI, Qdrant)")

    yield

    await db_service.close()
    logger.info("Stopped Multimodal PDF RAG API")


app = FastAPI(
    title="Multimodal PDF RAG API",
    description="FastAPI streaming multimodal RAG for one PDF source",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.cors_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/index")
async def index_pdf():
    service = create_indexing_service()
    try:
        result = await service.build_index()
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"status": "ok", **result}


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="message cannot be empty")

    logger.info(
        "chat_stream request received | incoming_thread_id=%s | message_chars=%d",
        request.thread_id or "(new)",
        len(message),
    )

    thread_id = request.thread_id or f"rag-thread-{uuid.uuid4().hex}"
    stats = get_qdrant().get_index_stats()
    logger.info(
        "chat_stream index stats | thread_id=%s | total_docs=%s | collection=%s",
        thread_id,
        stats.get("total_docs"),
        stats.get("collection_name"),
    )
    if int(stats.get("total_docs") or 0) == 0:
        logger.info("chat_stream blocked: no indexed docs | thread_id=%s", thread_id)

        async def not_indexed_stream():
            yield (
                f'data: {{"type":"session","thread_id":"{thread_id}"}}\n\n'
            )
            yield (
                'data: {"type":"end_of_response","thread_id":"'
                + thread_id
                + '","content":"Data is not indexed yet. Please call POST /index first.","sources":[]}\n\n'
            )

        return StreamingResponse(
            not_indexed_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream; charset=utf-8",
            },
        )

    logger.info("chat_stream creating ChatService | thread_id=%s", thread_id)
    service = create_chat_service()
    logger.info("chat_stream streaming start | thread_id=%s", thread_id)

    return StreamingResponse(
        service.stream_chat(message=message, thread_id=thread_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream; charset=utf-8",
        },
    )


@app.get("/index/stats")
async def get_index_stats():
    stats = get_qdrant().get_index_stats()
    return {"status": "ok", **stats}


@app.get("/index/pages/{page_number}")
async def get_index_page(page_number: int):
    if page_number <= 0:
        raise HTTPException(status_code=400, detail="page_number must be >= 1")
    page = get_qdrant().get_page_document(page_number=page_number)
    if page is None:
        raise HTTPException(status_code=404, detail=f"No indexed document found for page {page_number}")
    metadata = page["metadata"]
    token_usage = {
        "input_tokens": int(metadata.get("input_tokens") or 0),
        "output_tokens": int(metadata.get("output_tokens") or 0),
        "total_tokens": int(metadata.get("total_tokens") or 0),
    }
    return {
        "status": "ok",
        "page_number": page["page_number"],
        "page_content": page["page_content"],
        "metadata": metadata,
        "token_usage": token_usage,
    }


@app.get("/web", response_class=HTMLResponse)
async def web_interface():
    return WEB_UI
