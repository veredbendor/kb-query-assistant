import os
import pandas as pd
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

CHROMA_PATH = "./chroma"
THREADS_PATH = "data/rag_ready_threads.csv"
MAX_DOC_LENGTH = 8192
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize Chroma client and embedding function
client = chromadb.PersistentClient(path=CHROMA_PATH)
embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY,
    model_name="text-embedding-ada-002"
)

def load_threads_to_vector_store():
    print("🔄 Loading support threads into Chroma...")

    df = pd.read_csv(THREADS_PATH).dropna(subset=["content"])
    df = df[df["content"].str.len() < MAX_DOC_LENGTH]

    ids = [str(i) for i in df["ticketId"]]
    documents = df["content"].astype(str).tolist()
    metadatas = df.apply(lambda row: {
        "start": str(row["createdTime_start"]),
        "end": str(row["createdTime_end"]),
        "type": "thread"
    }, axis=1).tolist()

    try:
        client.delete_collection("support_threads")
    except:
        pass

    collection = client.create_collection(name="support_threads", embedding_function=embedding_fn)
    collection.add(ids=ids, documents=documents, metadatas=metadatas)

    print(f"✅ Loaded {len(ids)} support threads into vector store.")

if __name__ == "__main__":
    load_threads_to_vector_store()
