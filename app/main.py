import os
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
import httpx
import logging
from dotenv import load_dotenv

# ===== Setup =====
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY").strip()
RAG_API_URL = os.getenv("RAG_API_URL", "").strip()
CHROMA_PATH = "./chroma"
KB_PATH = "data/clean_kb_articles.csv"
THREADS_PATH = "data/rag_ready_threads.csv"
MAX_DOC_LENGTH = 8192
MAX_CONTEXT_CHARS = 6000

# ===== App Init =====
app = FastAPI(title="Zur Institute KB Assistant")

# ===== Chroma + Embedding =====
client = chromadb.PersistentClient(CHROMA_PATH)
embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY,
    model_name="text-embedding-ada-002"
)

def load_kb_collection():
    try:
        return client.get_collection(name="kb_articles", embedding_function=embedding_fn)
    except:
        df = pd.read_csv(KB_PATH).dropna(subset=["Answer"])
        ids = [str(i) for i in df["Article Id"].tolist()]
        documents = df["Answer"].tolist()
        metadatas = df.apply(lambda row: {
            "title": str(row["Article Title"]),
            "topic": str(row["Topic"]),
            "type": "article"
        }, axis=1).tolist()
        collection = client.create_collection(name="kb_articles", embedding_function=embedding_fn)
        collection.add(ids=ids, documents=documents, metadatas=metadatas)
        return collection

def load_thread_collection():
    try:
        return client.get_collection(name="support_threads", embedding_function=embedding_fn)
    except:
        df = pd.read_csv(THREADS_PATH).dropna(subset=["content"])
        df = df[df["content"].str.len() < MAX_DOC_LENGTH]  # Avoid large docs
        ids = [str(i) for i in df["ticketId"].tolist()]
        documents = df["content"].tolist()
        metadatas = df.apply(lambda row: {
            "start": str(row["createdTime_start"]),
            "end": str(row["createdTime_end"]),
            "type": "thread"
        }, axis=1).tolist()
        collection = client.create_collection(name="support_threads", embedding_function=embedding_fn)
        collection.add(ids=ids, documents=documents, metadatas=metadatas)
        return collection

kb_collection = load_kb_collection()
thread_collection = load_thread_collection()

# ===== Models =====
class QueryRequest(BaseModel):
    question: str
    top_k_articles: int = Field(3, ge=1, le=10)
    top_k_threads: int = Field(1, ge=0, le=10)

class ComposeRequest(BaseModel):
    description: str
    top_k_articles: int = Field(3, ge=1, le=10)
    top_k_threads: int = Field(1, ge=0, le=10)

# ===== Helpers =====
def build_context_from_results(article_results, thread_results, max_chars=MAX_CONTEXT_CHARS):
    context = ""
    for i in range(len(article_results["ids"][0])):
        meta = article_results["metadatas"][0][i]
        doc = article_results["documents"][0][i]
        context += f"[Article] Title: {meta['title']}\nAnswer: {doc}\n\n"
    for i in range(len(thread_results["ids"][0])):
        meta = thread_results["metadatas"][0][i]
        doc = thread_results["documents"][0][i]
        context += f"[Thread] From {meta['start']} to {meta['end']}\n{doc}\n\n"
    return context[:max_chars].strip()

# ===== Routes =====
@app.get("/")
def health():
    return {"status": "KB Assistant is live"}

@app.post("/query")
async def query_kb(req: QueryRequest):
    try:
        kb_results = kb_collection.query(query_texts=[req.question], n_results=req.top_k_articles)
        thread_results = thread_collection.query(query_texts=[req.question], n_results=req.top_k_threads)
        context = build_context_from_results(kb_results, thread_results)
        return {"context": context}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.post("/compose-reply")
async def compose_reply(request: ComposeRequest):
    try:
        client = AsyncOpenAI(api_key=OPENAI_API_KEY)

        kb_results = kb_collection.query(query_texts=[request.description], n_results=request.top_k_articles)
        thread_results = thread_collection.query(query_texts=[request.description], n_results=request.top_k_threads)
        rag_context = build_context_from_results(kb_results, thread_results)

        system_prompt = (
            "You are a professional and helpful customer support assistant for Zur Institute. "
            "Use the following knowledge base articles and support threads to compose a clear, concise, and helpful reply to the user's question. "
            "Summarize, rephrase, or combine relevant information. Do not copy as-is. Be friendly and professional.\n\n"
            f"Context:\n{rag_context}"
        )

        completion = await client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request.description}
            ]
        )
        return {"reply": completion.choices[0].message.content.strip()}

    except Exception as e:
        logger.exception("Unexpected error in compose_reply")
        return {"reply": f"Server error: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)