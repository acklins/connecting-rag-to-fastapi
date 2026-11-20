from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
import chromadb
import requests
import os

app = FastAPI(title="RAG API", version="1.0")

#CORSMiddleware — allow all origins, methods, headers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=False,
    allow_methods=["*"], 
    allow_headers=["*"],
)

# ── Config ─────────────────────────────
OLLAMA_URL = "http://localhost:11434"
MODEL = "llama3.2:1b"
DB_PATH = "./rag_db"
DOCS_DIR = "./docs"

# ── ChromaDB setup ───────────────────
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_or_create_collection(
    "documents", configuration={"hnsw": {"space": "cosine"}}
)

# ── Pydantic schemas ───────────────
class AskRequest(BaseModel):
    question: str
    n_results: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Number of document chunks to retrieve"
    )
    max_distance: float = Field(
        default=0.8,
        gt=0,
        le=2.0,
        description="Maximum cosine distance for retrieved chunks"
    )

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        question = value.strip()

        if not question:
            raise ValueError("Question cannot be empty")
        
        return question


class SourceChunk(BaseModel):
    text: str
    source: str
    distance: float


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    confidence: str


class IngestResponse(BaseModel):
    chunks_ingested: int
    message: str


# ── RAG helpers ─────────────────────
SYSTEM_PROMPT = ("""
You are a document-based question-answering assistant.

Your task is to answer questions accurately using the provided CONTEXT.

RULES:

1. If the context contains information that answers the question,
   provide a concise answer using that information.

2. You may summarize or paraphrase the context, but do not introduce
   facts, names, dates, examples, or claims absent from the context.

3. If the context only partially answers the question, provide the
   supported information and clearly identify what is missing.

4. If the context contains no information that answers the question,
   respond exactly:
   I don't know based on the provided documents.

5. Cite the source filename for factual answers.
   Example: (fastapi.txt)

6. Never cite documents that are not present in the context.

7. Keep answers under 100 words.

IMPORTANT:
Do not refuse to answer when the context explicitly contains
the answer. Use the available evidence.

The context includes source filenames in [Source: filename] labels.
"""
)

#Load text files and split them into paragraphs chunks.
def load_documents(directory: str) -> list[dict]:

    documents =[]

    if not os.path.isdir(directory):
        return documents

    for filename in sorted(os.listdir(directory)):
        if not filename.endswith(".txt"):
            continue

        filepath = os.path.join(directory, filename)

        with open(filepath, "r", encoding="utf-8") as file:
            content = file.read()

        paragraphs = [
            paragraph.strip()
            for paragraph in content.split("\n\n")
            if paragraph.strip()
        ]

        for index, paragraph in enumerate(paragraphs):
            documents.append({
                "id": f"{filename}_{index}",
                "text": paragraph,
                "metadata": {"source": filename}
            })
    return documents

#Query ChromaDB, filter by max_distance, return chunk dicts.
def retrieve(
        query: str, 
        n_results: int = 3, 
        max_distance: float = 0.8
) -> list[dict]:

    if collection.count() == 0:
        return []

    results = collection.query(
        query_texts=[query],
        n_results=min(n_results, collection.count()),
        include=["documents", "metadatas", "distances"]
    )

    chunks = []

    for text, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]

    ):
        if distance < max_distance:
            chunks.append({
                "text": text,
                "metadata": metadata,
                "distance": distance
            })
    return chunks

#Calculate retrieval confidence using the best distance.
def compute_confidence(chunks: list[dict]) -> str:

    if not chunks:
        return "low"

    best_distance = min(chunk["distance"]for chunk in chunks)

    if best_distance < 0.5:
        return "high"
    elif best_distance < 1.0:
        return "medium"
    else:
        return "low"

#Call Ollama and return its generated response
def call_ollama(messages: list[dict]) -> str:

    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.1}
    }
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json=payload,
            timeout=120
        )

        response.raise_for_status()

        data = response.json()
        return data["message"]["content"].strip()

    except requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=503,
            detail="Ollama is not running or cannot be reached."
        )

    except requests.exceptions.Timeout:
        raise HTTPException(
            status_code=503,
            detail="Ollama did not respond within 120 seconds."

        )

    except requests.exceptions.RequestException as error:
        raise HTTPException(
            status_code=502,
            detail=f"Ollama request failed: {error}"
        )

    except (ValueError, KeyError, TypeError):
        raise HTTPException(
            status_code=502,
            detail="Ollama returned an invalid response."
        )


#Check whether the Ollama server is reachable.
def check_ollama_health() -> bool:

    try:
        response = requests.get(
            f"{OLLAMA_URL}/api/tags",
            timeout=5
        )

        return response.status_code == 200

    except requests.exceptions.RequestException:
        return False


# ── Endpoints ──
#Retrieve context and generate a grounded answer.
@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):

    #Guardrail: handle an empty document collection.
    if collection.count() == 0:
        return AskResponse(
            answer="No documents have been ingested yet. RUN POST /ingest first. ",
            sources=[],
            confidence="low"
        )
    #Retrieve relevant document chunks.
    chunks = retrieve(
        query=req.question,
        n_results=req.n_results,
        max_distance=req.max_distance
    )

    #Guardrail: no chunks passed the distance threshold.
    if not chunks:
        return AskResponse(
            answer="I couldn't find relevant info in the documents.",
            sources=[],
            confident="low"
        )

    #Calculate retrival confidence.
    confidence = compute_confidence(chunks)

    #Build context with source labels.
    context_parts = []

    for chunk in chunks:
        source = chunk["metadata"]["source"]
        text = chunk["text"]

        context_parts.append(
            f"[Source: {source}]\n{text}"
        )
    context = "\n\n---\n\n".join(context_parts)

    #Send the context and question to Ollama.
    messages = [
        {
            "role": "system",
            "content": f"{SYSTEM_PROMPT}\n\nCONTEXT:\n{context}"
        },
        {
            "role": "user",
            "content": req.question
        }
    ]

    answer = call_ollama(messages)

    #Build structured source objects.
    sources = [
        SourceChunk(
            text=chunk["text"],
            source=chunk["metadata"]["source"],
            distance=chunk["distance"]
        )
        for chunk in chunks
    ]

    return AskResponse(
        answer=answer,
        sources=sources,
        confidence=confidence
    )


#Load documents from docs/ into ChromaDB.
@app.post("/ingest", response_model=IngestResponse)
def ingest():

    documents = load_documents(DOCS_DIR)

    if not documents:
        raise HTTPException(
            status_code=404,
            detail="No text documents found in the docs/folder."
        )

    try: 
        collection.upsert(
            ids=[doc["id"] for doc in documents],
            documents=[doc["text"] for doc in documents],
            metadatas=[doc["metadata"] for doc in documents]
        )
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Document ingestion failed: {error}"
        )

    return IngestResponse(
        chunks_ingested=len(documents),
        message=f"Succesfully ingested {len(documents)} document chunks."
    )

#Pydantic response model
class StatsResponse(BaseModel):
    document_count: int
    model: str
    db_path: str

class HealthResponse(BaseModel):
    status: str
    chromadb: str
    ollama: str
    document_count: int

#Return document count and model info
@app.get("/stats", response_model=StatsResponse)
def stats():

    try:
        count = collection.count()

    except Exception:
        raise HTTPException(
            status_code=503,
            detail="ChromaDB is unavailable."
        )

    return StatsResponse(
        document_count=count,
        model=MODEL,
        db_path=DB_PATH
    )

#Check ChromaDB and Ollama liveness.
@app.get("/health")
def health():

    try:
        document_count = collection.count()
        chromadb_status = "ok"

    except Exception:
        document_count = 0
        chromadb_status = "error"

    ollama_status = (
        "connected" if check_ollama_health()
        else "disconnected"
    )

    overall_status = (
        "ok"
        if chromadb_status == "ok"
        and ollama_status == "connected"
        else "degraded"
    )

    return HealthResponse(
        status=overall_status,
        chromadb=chromadb_status,
        ollama=ollama_status,
        document_count=document_count
    )