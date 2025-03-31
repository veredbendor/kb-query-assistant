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
async def compose_reply(request: ComposeRequest):
    try:
        # Create a fresh client per request
        client = AsyncOpenAI(api_key=OPENAI_API_KEY)

        # 1. Query RAG API
        try:
            async with httpx.AsyncClient() as http_client:
                rag_response = await http_client.post(RAG_API_URL, json={
                    "question": request.description,
                    "top_k": request.top_k
                })
            rag_response.raise_for_status()
            rag_context = rag_response.json().get("context", "")
        except Exception as e:
            logger.error(f"RAG API error: {str(e)}")
            return {"reply": f"Error retrieving context: {str(e)}"}

        # 2. Call OpenAI ChatCompletion
        try:
            system_prompt = (
                "You are a professional and helpful customer support assistant for Zur Institute. "
                "Use the following knowledge base articles to compose a clear, concise, and helpful reply to the user's question. "
                "Summarize, rephrase, or combine relevant information from the articles. "
                "Do not copy the articles as-is. Be friendly and professional.\n\n"
                "Knowledge Base Context:\n"
                f"{rag_context}"
            )
            completion = await client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": request.description}
                ]
            )
            ai_reply = completion.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            return {"reply": f"Error generating response: {str(e)}"}

        return {"reply": ai_reply}
    except Exception as e:
        logger.exception("Unexpected error in compose_reply")
        return {"reply": f"Server error: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))  # Railway injects PORT
    uvicorn.run(app, host="0.0.0.0", port=port)

