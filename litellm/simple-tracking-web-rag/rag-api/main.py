"""
RAG API – OpenAI-compatible /v1/chat/completions endpoint
Exposes the RAG pipeline as a drop-in model that LiteLLM can proxy.
"""

import os, uuid, json, logging
from typing import AsyncIterator, List, Optional

import httpx
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Config from env ────────────────────────────────────────────────────────────
QDRANT_HOST      = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT      = int(os.getenv("QDRANT_PORT", "6333"))
LITELLM_BASE_URL = os.getenv("LITELLM_BASE_URL", "http://host.docker.internal:4000/v1")
LITELLM_API_KEY  = os.getenv("LITELLM_API_KEY", "sk-1234")
EMBEDDING_MODEL  = os.getenv("EMBEDDING_MODEL", "text-embedding")
CHAT_MODEL       = os.getenv("CHAT_MODEL", "llama3")
COLLECTION       = os.getenv("QDRANT_COLLECTION", "documents")
VECTOR_SIZE      = int(os.getenv("VECTOR_SIZE", "768"))
TOP_K            = int(os.getenv("TOP_K", "5"))
SCORE_THRESHOLD  = float(os.getenv("SCORE_THRESHOLD", "0.3"))

# ── Clients ────────────────────────────────────────────────────────────────────
qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
http   = httpx.AsyncClient(timeout=120)

app = FastAPI(title="RAG API", version="1.0.0")


# ── Ensure collection exists ───────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    existing = [c.name for c in qdrant.get_collections().collections]
    if COLLECTION in existing:
        info = qdrant.get_collection(COLLECTION)
        actual_size = info.config.params.vectors.size
        if actual_size != VECTOR_SIZE:
            logger.warning(
                f"Collection '{COLLECTION}' has dim={actual_size}, "
                f"need dim={VECTOR_SIZE}. Recreating (old data will be lost)."
            )
            qdrant.delete_collection(COLLECTION)
            existing.remove(COLLECTION)

    if COLLECTION not in existing:
        qdrant.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        logger.info(f"Created Qdrant collection '{COLLECTION}' with dim={VECTOR_SIZE}")


# ── Helpers ────────────────────────────────────────────────────────────────────
async def embed(texts: List[str]) -> List[List[float]]:
    """Call LiteLLM embeddings endpoint."""
    r = await http.post(
        f"{LITELLM_BASE_URL}/embeddings",
        headers={"Authorization": f"Bearer {LITELLM_API_KEY}"},
        json={"model": EMBEDDING_MODEL, "input": texts},
    )
    r.raise_for_status()
    return [d["embedding"] for d in r.json()["data"]]


async def retrieve(query: str, collection: str = COLLECTION, top_k: int = TOP_K):
    """Embed query → search Qdrant → return top chunks above score threshold."""
    [q_vec] = await embed([query])
    hits = qdrant.search(
        collection_name=collection,
        query_vector=q_vec,
        limit=top_k,
        with_payload=True,
        score_threshold=SCORE_THRESHOLD,
    )
    logger.info(f"Retrieval scores: {[round(h.score, 3) for h in hits]}")
    return [h.payload for h in hits], [h.score for h in hits]


def build_rag_messages(context_chunks: list, scores: list, messages: list) -> list:
    """Prepend retrieved context as a system message, with source attribution."""
    if not context_chunks:
        # No relevant context found — tell the model explicitly
        context_text = "No relevant context was found in the knowledge base."
    else:
        parts = []
        for i, (chunk, score) in enumerate(zip(context_chunks, scores), 1):
            source = chunk.get("source", "unknown")
            text   = chunk.get("text", "")
            parts.append(f"[{i}] (source: {source}, relevance: {score:.2f})\n{text}")
        context_text = "\n\n---\n\n".join(parts)

    rag_system = (
        "You are a helpful assistant with access to a knowledge base.\n"
        "Use the CONTEXT below to answer the user's question.\n"
        "Always cite which context chunk(s) you used, e.g. [1], [2].\n"
        "If the answer is not in the context, say so clearly — do not make things up.\n\n"
        f"CONTEXT:\n{context_text}"
    )

    result = [m for m in messages if m["role"] != "system"]
    return [{"role": "system", "content": rag_system}] + result


# ── OpenAI-compatible schemas ──────────────────────────────────────────────────
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str = "rag"
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1024
    stream: Optional[bool] = False
    collection: Optional[str] = None
    top_k: Optional[int] = None


# ── Streaming helper ───────────────────────────────────────────────────────────
async def stream_from_litellm(payload: dict) -> AsyncIterator[str]:
    """Forward a streaming request to LiteLLM and yield SSE chunks."""
    async with http.stream(
        "POST",
        f"{LITELLM_BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {LITELLM_API_KEY}"},
        json={**payload, "stream": True},
        timeout=120,
    ) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if line:
                yield f"{line}\n\n"


# ── /v1/chat/completions ───────────────────────────────────────────────────────
@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    user_messages = [m for m in req.messages if m.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="No user message found")
    query = user_messages[-1].content

    collection = req.collection or COLLECTION
    top_k      = req.top_k or TOP_K

    # 1. Retrieve
    chunks, scores = await retrieve(query, collection=collection, top_k=top_k)
    logger.info(f"Retrieved {len(chunks)} chunks above threshold={SCORE_THRESHOLD} for: {query[:80]}")

    # 2. Augment
    augmented_messages = build_rag_messages(chunks, scores, [m.dict() for m in req.messages])

    litellm_payload = {
        "model": CHAT_MODEL,
        "messages": augmented_messages,
        "temperature": req.temperature,
        "max_tokens": req.max_tokens,
    }

    # 3a. Streaming — what Open WebUI uses by default
    if req.stream:
        return StreamingResponse(
            stream_from_litellm(litellm_payload),
            media_type="text/event-stream",
        )

    # 3b. Non-streaming
    r = await http.post(
        f"{LITELLM_BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {LITELLM_API_KEY}"},
        json=litellm_payload,
    )
    r.raise_for_status()
    result = r.json()
    result["model"] = req.model
    return result


# ── Document ingestion ─────────────────────────────────────────────────────────
class IngestRequest(BaseModel):
    texts: List[str]
    metadatas: Optional[List[dict]] = None
    collection: Optional[str] = None


@app.post("/v1/ingest", summary="Ingest text chunks into the vector store")
async def ingest(req: IngestRequest):
    collection = req.collection or COLLECTION
    metadatas  = req.metadatas or [{} for _ in req.texts]

    if len(metadatas) != len(req.texts):
        raise HTTPException(status_code=400, detail="texts and metadatas must be same length")

    embeddings = await embed(req.texts)
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=emb,
            payload={"text": txt, **meta},
        )
        for emb, txt, meta in zip(embeddings, req.texts, metadatas)
    ]
    qdrant.upsert(collection_name=collection, points=points)
    return {"ingested": len(points), "collection": collection}


@app.post("/v1/ingest/file", summary="Upload and ingest a plain-text or .md file")
async def ingest_file(
    file: UploadFile = File(...),
    chunk_size: int  = Form(500),
    collection: str  = Form(COLLECTION),
):
    content = (await file.read()).decode("utf-8", errors="ignore")
    chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
    chunks = [c.strip() for c in chunks if c.strip()]

    embeddings = await embed(chunks)
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=emb,
            payload={"text": txt, "source": file.filename},
        )
        for emb, txt in zip(embeddings, chunks)
    ]
    qdrant.upsert(collection_name=collection, points=points)
    return {"ingested": len(points), "collection": collection, "source": file.filename}


# ── Collection management ──────────────────────────────────────────────────────
@app.get("/v1/collections", summary="List all vector collections")
async def list_collections():
    cols = qdrant.get_collections().collections
    return {"collections": [c.name for c in cols]}


@app.delete("/v1/collections/{name}", summary="Delete a collection")
async def delete_collection(name: str):
    qdrant.delete_collection(name)
    return {"deleted": name}


# ── Health ─────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok"}


# ── Model listing ──────────────────────────────────────────────────────────────
@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{"id": "rag", "object": "model", "owned_by": "local"}],
    }
