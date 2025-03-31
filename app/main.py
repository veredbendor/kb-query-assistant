import os
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from openai import AsyncOpenAI, OpenAIError
import httpx
import logging
from dotenv import load_dotenv

# ===== Setup =====

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY").strip()
CHROMA_PATH = "./chroma"
DATA_PATH = "data/clean_kb_articles.csv"

# ===== App Init =====

app = FastAPI(title="Zur Institute KB Assistant")

# ===== Chroma + Embedding =====

client = chromadb.PersistentClient(CHROMA_PATH)
embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY,
    model_name="text-embedding-ada-002"
)

def load_collection():
    try:
        collection = client.get_collection(name="kb_articles", embedding_function=embedding_fn)
        return collection
    except Exception:
        # Prepare collection
        collection = client.create_collection(name="kb_articles", embedding_function=embedding_fn)
        df = pd.read_csv(DATA_PATH)
        df = df.dropna(subset=["Answer"])  # Clean NaN
        ids = [str(i) for i in df["Article Id"].tolist()]
        documents = df["Answer"].tolist()
        metadatas = df.apply(lambda row: {
            "title": str(row["Article Title"]),
            "topic": str(row["Topic"]),
        }, axis=1).tolist()
        collection.add(ids=ids, documents=documents, metadatas=metadatas)
        return collection

collection = load_collection()

# ===== Models =====

class QueryRequest(BaseModel):
    question: str
    top_k: int = Field(2, ge=1, le=10)

class ComposeRequest(BaseModel):
    description: str
    top_k: int = Field(2, ge=1, le=10)

# ===== Routes =====

@app.get("/")
def health():
    return {"status": "KB Assistant is live"}

@app.post("/query")
async def query_kb(req: QueryRequest):
    try:
        results = collection.query(
            query_texts=[req.question],
            n_results=req.top_k
        )
        context = ""
        for i in range(len(results["ids"][0])):
            title = results["metadatas"][0][i]["title"]
            answer = results["documents"][0][i]
            context += f"Title: {title}\nAnswer: {answer}\n\n"
        return {"context": context.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.post("/compose-reply")
async def compose_reply(req: ComposeRequest):
    try:
        # Step 1 - Query RAG
        results = collection.query(
            query_texts=[req.description],
            n_results=req.top_k
        )
        context = ""
        for i in range(len(results["ids"][0])):
            title = results["metadatas"][0][i]["title"]
            answer = results["documents"][0][i]
            context += f"Title: {title}\nAnswer: {answer}\n\n"

        # Step 2 - Compose reply
        client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        system_prompt = (
            "You are a customer support assistant for Zur Institute. "
            "Use the following knowledge base context when answering:\n"
            f"{context}"
        )
        completion = await client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": req.description}
            ]
        )
        reply = completion.choices[0].message.content.strip()
        return {"reply": reply}

    except OpenAIError as e:
        raise HTTPException(status_code=500, detail=f"Compose failed: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Compose failed: {str(e)}")

# ====== Run locally ======
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
