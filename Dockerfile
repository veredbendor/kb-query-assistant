FROM python:3.9-slim

WORKDIR /app

# Copy requirements first
COPY requirements.txt .

# Install dependencies with verbose output
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir pandas openai fastapi uvicorn python-dotenv chromadb

# Copy the rest of the application
COPY . .

# Create necessary directories
RUN mkdir -p chroma

# Simple command to run the app
CMD ["python", "app/main.py"]
