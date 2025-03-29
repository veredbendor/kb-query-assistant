from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import os
import openai
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
from chromadb.errors import InvalidCollectionException
from typing import List, Dict, Any

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

# Data model for query requests
class QueryRequest(BaseModel):
    question: str
    top_k: int = 3  # Default to top 3 results

# Data model for query responses
class QueryResponse(BaseModel):
    matches: List[Dict[str, Any]]
    
# Initialize collection
def initialize_collection():
    try:
        # Try to get existing collection
        collection = client.get_collection(name="kb_articles", embedding_function=openai_ef)
        print("Found existing collection with", collection.count(), "documents")
        return collection
    except Exception as e:
        print(f"Collection error: {e}")
        print("Creating new collection...")
        
        # Create new collection
        collection = client.create_collection(name="kb_articles", embedding_function=openai_ef)
        
        try:
            # Load KB articles
            csv_path = "data/clean_kb_articles.csv"
            print(f"Loading data from {csv_path}")
            df = pd.read_csv(csv_path)
            print(f"Loaded {len(df)} rows")
            
            # Prepare data for Chroma
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
            
            # Add documents to collection in batches
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

# Initialize collection
collection = initialize_collection()

@app.get("/")
def read_root():
    return {"message": "Knowledge Base Query Assistant API", "status": "online"}

@app.post("/query", response_model=QueryResponse)
async def query_kb(query_request: QueryRequest):
    try:
        # Query the collection
        results = collection.query(
            query_texts=[query_request.question],
            n_results=query_request.top_k
        )
        
        # Format the response
        matches = []
        for i in range(len(results["ids"][0])):
            matches.append({
                "article_id": results["ids"][0][i],
                "title": results["metadatas"][0][i]["title"],
                "answer": results["documents"][0][i],
                "topic": results["metadatas"][0][i]["topic"],
                "similarity": results["distances"][0][i] if "distances" in results else None
            })
        
        return {"matches": matches}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

# Run directly if main
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)