from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import pandas as pd
import os
import openai
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Any
import re

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize FastAPI app
app = FastAPI(title="Knowledge Base Query Assistant")

# Initialize ChromaDB client
client = chromadb.PersistentClient("./chroma")

# Initialize OpenAI embedding function
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=openai.api_key,
    model_name="text-embedding-ada-002"
)

# Initialize collection
def initialize_collection():
    try:
        collection = client.get_collection(name="kb_articles", embedding_function=openai_ef)
        print("Found existing collection with", collection.count(), "documents")
        return collection
    except Exception as e:
        print(f"Collection error: {e}")
        print("Creating new collection...")

        collection = client.create_collection(name="kb_articles", embedding_function=openai_ef)
        try:
            csv_path = "data/clean_kb_articles.csv"
            print(f"Loading data from {csv_path}")
            df = pd.read_csv(csv_path)
            print(f"Loaded {len(df)} rows")

            ids = [str(i) for i in df["Article Id"].tolist()]
            documents = df["Answer"].tolist()
            metadatas = df.apply(
                lambda row: {
                    "title": str(row["Article Title"]),
                    "topic": str(row["Topic"]),
                    "public": bool(row["Public"]),
                    "created": str(row["Created Time"]),
                    "modified": str(row["Modified Time"])
                },
                axis=1
            ).tolist()

            batch_size = 100
            for i in range(0, len(ids), batch_size):
                end_idx = min(i + batch_size, len(ids))
                collection.add(
                    ids=ids[i:end_idx],
                    documents=documents[i:end_idx],
                    metadatas=metadatas[i:end_idx]
                )
                print(f"Added batch {i//batch_size + 1}, documents {i} to {end_idx}")

            return collection

        except Exception as e:
            print(f"Error loading KB data: {e}")
            print("Created empty collection for testing")
            return collection

# Initialize
collection = initialize_collection()

# Models
class QueryRequest(BaseModel):
    question: str
    top_k: int = 3

class ComposeRequest(BaseModel):
    description: str
    top_k: int = 2

@app.get("/")
def read_root():
    return {"message": "Knowledge Base Query Assistant API", "status": "online"}

@app.post("/query")
async def query_kb(query_request: QueryRequest):
    try:
        results = collection.query(
            query_texts=[query_request.question],
            n_results=query_request.top_k
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
        # Step 1: Mask PII
        description = req.description
        description = re.sub(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", "[EMAIL]", description)
        description = re.sub(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b", "[PHONE]", description)
        description = re.sub(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b", "[NAME]", description)

        # Step 2: Query KB
        results = collection.query(
            query_texts=[description],
            n_results=req.top_k
        )
        context = ""
        for i in range(len(results["ids"][0])):
            title = results["metadatas"][0][i]["title"]
            answer = results["documents"][0][i]
            context += f"Title: {title}\nAnswer: {answer}\n\n"

        # Step 3: Compose with OpenAI
        system_prompt = f"You are a customer support assistant for Zur Institute. Use the following knowledge base context when answering.\nContext:\n{context.strip()}"

        completion = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": description}
            ]
        )

        reply = completion["choices"][0]["message"]["content"]
        return {"reply": reply}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Compose failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
