import os
import requests
import pandas as pd
from dotenv import load_dotenv

import chromadb
from chromadb import PersistentClient 
from chromadb.config import Settings
from chromadb.utils import embedding_functions

# Load env vars
load_dotenv()

CLIENT_ID = os.getenv("ZOHO_CLIENT_ID")
CLIENT_SECRET = os.getenv("ZOHO_CLIENT_SECRET")
REFRESH_TOKEN = os.getenv("ZOHO_REFRESH_TOKEN")
ORG_ID = os.getenv("ZOHO_ORG_ID")
CSV_OUTPUT_PATH = "latest_kb_articles.csv"

# Step 1: Get new access token
def get_access_token():
    response = requests.post(
        "https://accounts.zoho.com/oauth/v2/token",
        params={
            "refresh_token": REFRESH_TOKEN,
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "grant_type": "refresh_token"
        }
    )
    response.raise_for_status()
    return response.json()["access_token"]

# Step 2: Fetch all paginated articles
def fetch_all_articles():
    token = get_access_token()
    headers = {
        "Authorization": f"Zoho-oauthtoken {token}",
        "orgId": ORG_ID
    }

    base_url = "https://desk.zoho.com/api/v1/articles"
    offset = 1  # Start at 1 (Zoho API seems to expect this)
    limit = 50
    all_articles = []

    while True:
        params = {
            "from": offset,
            "limit": limit
        }
        response = requests.get(base_url, headers=headers, params=params)

        if response.status_code == 422:
            print("❌ 422 error: check docs for required params or max page size.")
            print("Response:", response.json())
            break

        response.raise_for_status()
        data = response.json().get("data", [])
        published_articles = [article for article in data if article.get("status") == "Published"]
        all_articles.extend(published_articles)

        print(f"✅ Fetched {len(published_articles)} published articles (offset {offset})")

        if not data or len(data) < limit:
            break  # Done paging

        offset += limit

    return all_articles





# Step 3: Extract useful fields and save to CSV
def save_articles_to_csv(articles):
    rows = []
    for article in articles:
        rows.append({
            "Article Id": article.get("id"),
            "Article Title": article.get("title"),
            "Answer": article.get("summary", ""),  # fallback to summary
            "Status": article.get("status"),
            "Created Time": article.get("createdTime"),
            "Modified Time": article.get("modifiedTime"),
            "URL": article.get("webUrl"),
            "Topic": article.get("category", {}).get("name")
        })

    df = pd.DataFrame(rows)
    df.to_csv(CSV_OUTPUT_PATH, index=False)
    print(f"✅ Saved {len(df)} articles to {CSV_OUTPUT_PATH}")
    
# Step 4: Update ChromaDB with new embeddings
def update_vector_store():
    print("🔄 Updating Chroma vector store with latest KB articles...")

    # Load cleaned CSV
    df = pd.read_csv(CSV_OUTPUT_PATH).dropna(subset=["Answer"])

    # Initialize Chroma PersistentClient
    client = PersistentClient(path="chroma")  # <-- no Settings() needed here

    # Drop old collection if it exists
    if "kb_articles" in [col.name for col in client.list_collections()]:
        client.delete_collection("kb_articles")

    # Recreate collection with embedding
    embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="text-embedding-ada-002"
    )
    collection = client.create_collection(name="kb_articles", embedding_function=embedding_fn)

    # Add articles
    collection.add(
        documents=df["Answer"].astype(str).tolist(),
        ids=[str(row["Article Id"]) for _, row in df.iterrows()],
        metadatas=[
            {
                "title": row["Article Title"],
                "topic": row.get("Topic", ""),
                "url": row.get("URL", "")
            }
            for _, row in df.iterrows()
        ]
    )

    print(f"✅ Replaced kb_articles collection with {len(df)} documents.")



# Run
if __name__ == "__main__":
    articles = fetch_all_articles()
    save_articles_to_csv(articles)
    update_vector_store()
