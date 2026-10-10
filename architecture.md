# Architecture

```text
Six FinBase PDFs
      |
      v
PDF extraction + cleaning + page metadata
      |
      v
Section/FAQ-aware chunking
      |
      v
TF-IDF vector embeddings (persisted locally)
      |
      v
Cosine-similarity vector retrieval (top-k)
      |
      +---- relevance threshold ----> Not-found response
      |
      v
Grounded prompt / extractive fallback
      |
      v
Answer + document/page citations
      |
      v
FastAPI JSON API + browser chat UI
```

## Why these choices

- **Python + FastAPI:** lightweight API with automatic OpenAPI documentation.
- **PDF extraction with pypdf:** deterministic and preserves page-level citation metadata.
- **Section/FAQ-aware chunking:** prevents unrelated policy clauses from being mixed and keeps FAQ answers coherent.
- **TF-IDF vector representations + cosine search:** fully local, reproducible, and does not require a hosted vector database or model download. The design can be upgraded to sentence-transformer embeddings/FAISS without changing the API contract.
- **Optional OpenAI LLM:** when `OPENAI_API_KEY` is supplied, the model receives only retrieved context and is explicitly instructed not to invent policy information. Without a key, an extractive fallback keeps the demo runnable.
- **Relevance threshold:** low-confidence retrieval produces a clear “not found” response instead of hallucinating.
- **Citations:** every grounded response returns document name, page, section, score, and a short evidence snippet.

## Hallucination mitigation

1. Retrieve before generation.
2. Reject low-similarity retrievals.
3. Pass only retrieved context to the LLM.
4. Tell the LLM to refuse unsupported claims.
5. Return evidence citations alongside the answer.
6. Test an out-of-domain question in the evaluation set.
