# 5-Minute Demo Script

## 0:00–0:35 — Problem
“Financial support questions often require exact policy details. This project builds a FinBase customer-support assistant that answers only from six supplied policy PDFs and cites the evidence used.”

## 0:35–1:20 — Architecture
Show `architecture.md`. Explain: PDFs → cleaning/page metadata → FAQ/section-aware chunks → vector representations → cosine retrieval → relevance guardrail → grounded answer → citations.

## 1:20–2:05 — Data and retrieval
Open `data/` and show the six PDFs. Explain that page numbers are retained so every answer can point back to a document page. Mention the local vector store keeps the demo reproducible without external infrastructure.

## 2:05–3:25 — Live demo
Ask:
1. “What is the foreclosure charge if I close my personal loan after 26 months?”
2. “What is the minimum balance required for a FinBase Digital Savings Account?”
3. Ask a follow-up such as “What about after 18 months?”
4. “What is the weather forecast for Delhi tomorrow?”

Point out that the first questions show evidence and page citations, while the last question is refused because it is not in the knowledge base.

## 3:25–4:10 — Evaluation
Run `python evaluation/evaluate.py`. Explain Hit@5, MRR, answer keyword accuracy, citation accuracy, grounded response rate, and not-found detection.

## 4:10–4:45 — Engineering choices
Explain why FastAPI was selected, why page-aware chunks matter, why a local vector store was used, and how an optional OpenAI model can phrase retrieved context without being allowed to invent policy facts.

## 4:45–5:00 — Trade-offs / next steps
Say that with more time you would add a production embedding model, reranking, a managed vector database, automated LLM-as-judge evaluation, authentication, observability, and a cloud deployment.
