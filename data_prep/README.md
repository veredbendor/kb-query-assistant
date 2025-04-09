# Knowledge Base Vector Prep

This script fetches the latest knowledge base articles from Zoho Desk using the API and updates the local ChromaDB vector store.

## What it does
- Downloads KB articles using Zoho OAuth token
- Normalizes and saves a clean CSV
- Replaces the `kb_articles` collection in ChromaDB with the latest content

## Usage
1. Set your environment variables in `.env`:
   - `ZOHO_ACCESS_TOKEN`
   - `ZOHO_ORG_ID`
   - `OPENAI_API_KEY`

2. Run:
   ```bash
   python data_prep/update_kb_vectors.py
