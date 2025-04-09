import pandas as pd
import chromadb
from chromadb.config import Settings

# Load the CSV
df = pd.read_csv("data/rag_ready_threads.csv")

# Initialize Chroma DB client (default settings for local dev)
chroma_client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory="chroma"))

# Create or get collection
collection = chroma_client.get_or_create_collection(name="zoho_support_threads")

# Add each document
for idx, row in df.iterrows():
    doc_id = str(row["ticketId"])
    content = row["content"]
    metadata = {
        "createdTime_start": row["createdTime_start"],
        "createdTime_end": row["createdTime_end"]
    }
    collection.add(
        documents=[content],
        ids=[doc_id],
        metadatas=[metadata]
    )

print(f"✅ Loaded {len(df)} documents into Chroma collection 'zoho_support_threads'")
