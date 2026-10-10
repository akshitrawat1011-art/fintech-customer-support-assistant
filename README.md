# FinBase Financial Support RAG Assistant

Submission-ready implementation for the AIML Engineering assignment: a document-grounded financial customer-support assistant built over the six supplied FinBase PDFs.

## What it demonstrates

- PDF loading and preprocessing
- Page-aware, FAQ/section-aware chunking
- Vector representations and cosine retrieval
- Grounded answer generation
- Source/document/page citations
- Explicit not-found behavior
- Follow-up questions using recent conversation context
- Evaluation for retrieval, answer keyword accuracy, groundedness and citation accuracy
- FastAPI API + browser chat interface
- Docker support
- Automatic OpenAPI docs at `/docs`

## Project structure

```text
app/
  main.py             # FastAPI API + browser UI
  rag.py              # ingestion, chunking, vector store, retrieval, answer pipeline
data/
  sample_1.pdf ... sample_6.pdf
evaluation/
  eval_set.json       # representative evaluation questions
  evaluate.py         # evaluation harness
scripts/
  build_index.py      # rebuild local index
storage/              # generated persisted index files
tests/
  test_rag.py
architecture.md
Dockerfile
docker-compose.yml
.env.example
```

## Run locally

Python 3.11+ is recommended.

```bash
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
python scripts/build_index.py
uvicorn app.main:app --reload --port 8080
```

Open `http://localhost:8080` for the chat UI.

API documentation: `http://localhost:8080/docs`

Health check: `http://localhost:8080/health`

## Optional LLM mode

Copy `.env.example` to `.env` and add an OpenAI API key. The system then uses the selected model to phrase the answer while keeping the retrieved policy text as the only factual context.

If no key is provided, the application still works using the deterministic extractive fallback. This is useful for testing and for environments where external model access is unavailable.

## Rebuild after changing PDFs

```bash
python scripts/build_index.py
```

## Evaluation

```bash
python evaluation/evaluate.py
```

The harness reports:

- Retrieval Hit@5
- Mean Reciprocal Rank@5
- Answer keyword accuracy
- Citation accuracy
- Grounded response rate
- Not-found detection

## Tests

```bash
pytest -q
```

## Docker

```bash
docker build -t finbase-rag .
docker run --rm -p 8080:8080 finbase-rag
```

Or:

```bash
docker compose up --build
```

## Demo questions

1. What is the foreclosure charge if I close my personal loan after 26 months?
2. What is the minimum balance required for a FinBase Digital Savings Account?
3. My UPI transaction failed but money was deducted. When will I receive a refund?
4. What documents are required to complete Video KYC verification?
5. What is the interest rate for senior citizens on a 1-year Fixed Deposit?
6. Does FinBase provide Bitcoin wallets?  
7. What is the weather forecast for Delhi tomorrow?  ← demonstrates the not-found guardrail.

## Assignment mapping

| Assignment requirement | Implementation |
|---|---|
| Load financial documents | `app/rag.py` PDF ingestion |
| Clean/preprocess | whitespace/currency cleanup + page metadata |
| Chunking | FAQ/section-aware page chunks |
| Embeddings/vector search | TF-IDF vectors + cosine similarity |
| Retrieval → context → answer | `answer_query()` |
| Grounded responses | relevance threshold + context-only LLM prompt |
| Source references | document + page + section + snippet |
| Follow-ups | recent user-turn context |
| Evaluation | `evaluation/evaluate.py` + `eval_set.json` |
| Interface | FastAPI-served browser UI |
| API documentation | FastAPI `/docs` |
| Dockerization | `Dockerfile`, `docker-compose.yml` |
| Environment variables | `.env.example` |
| Architecture explanation | `architecture.md` |
