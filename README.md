# Knowledge Base Query Assistant

A simple Docker-based MVP for querying knowledge base articles using vector embeddings and semantic search.

## Project Structure

```
kb-query-assistant/
├── data/
│   └── clean_kb_articles.csv     # Cleaned knowledge base articles
├── app/
│   └── main.py                   # FastAPI app + all functionality
├── chroma/                       # ChromaDB files (auto-created)
├── requirements.txt              # Dependencies
├── Dockerfile                    # For containerizing the application
├── railway.json                  # Railway deployment configuration
├── .env                          # API keys (create from .env.example)
└── README.md                     # This file
```

## Docker Setup & Run (Recommended)

The easiest way to run this application is with Docker:

1. Make sure Docker is installed on your system
2. Clone this repository
3. Place your knowledge base CSV in the `data/` directory as `clean_kb_articles.csv`
4. Create a `.env` file with your OpenAI API key
5. Build and run the container:

```bash
# Build the Docker image
docker build -t kb-query-assistant .

# Run the container
docker run -p 8000:8000 --env-file .env -v $(pwd)/data:/app/data -v $(pwd)/chroma:/app/chroma kb-query-assistant
```

The volume mounts ensure that:
- Your data file is accessible to the container
- The ChromaDB files persist between container restarts

## API Usage

Once the container is running, you can access:

- Swagger UI: http://localhost:8000/docs
- API endpoint: http://localhost:8000/query

Example query with curl:
```bash
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"question":"How do I print to PDF?", "top_k": 2}'
```

## Railway Deployment

For production deployment:

1. Push your code to GitHub
2. Link your Railway account to your GitHub repository
3. Create a new project in Railway from the repository
4. Add your OPENAI_API_KEY as an environment variable
5. Deploy the service

Railway will automatically use the `railway.json` and `Dockerfile` to build and deploy your application.

## Local Development Setup (Alternative)

If you prefer to develop without Docker:

1. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up your environment variables:
   ```
   cp .env.example .env
   # Then edit .env with your editor
   ```

4. Run the application:
   ```
   uvicorn app.main:app --reload
   ```

## How It Works

1. The application loads articles from your CSV file
2. It generates embeddings using OpenAI's API
3. Embeddings are stored in a ChromaDB vector database
4. When a query is received, it finds semantically similar content
5. The most relevant articles are returned as responses