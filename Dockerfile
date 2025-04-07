FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy's large English model required by presidio-analyzer for high-accuracy NER
# presidio-analyzer and presidio-anonymizer are used to detect and redact sensitive PII like names, emails, and license IDs
# They integrate with spaCy to improve entity detection in unstructured support ticket data
# These are already in requirements.txt, so no extra step is needed here unless you want to verify model availability
RUN python -m spacy download en_core_web_lg


COPY . .
RUN mkdir -p chroma

EXPOSE 8001

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
