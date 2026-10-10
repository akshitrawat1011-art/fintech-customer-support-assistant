# Submission Checklist

- [x] Complete source code
- [x] Six supplied financial PDFs included in `data/`
- [x] PDF preprocessing and page metadata
- [x] Chunking strategy
- [x] Vector representations and retrieval
- [x] Grounded response pipeline
- [x] Source/document/page citations
- [x] Not-found guardrail
- [x] Follow-up conversation support
- [x] Working browser UI
- [x] FastAPI API + `/docs`
- [x] Evaluation dataset
- [x] Retrieval/correctness/groundedness/citation metrics
- [x] Unit tests
- [x] README and environment documentation
- [x] Architecture explanation
- [x] Dockerfile and docker-compose
- [x] 5-minute demo script
- [ ] Deploy to a public URL
- [ ] Record the 5-minute video

## Final local verification

```bash
python scripts/build_index.py
pytest -q
python evaluation/evaluate.py
uvicorn app.main:app --reload --port 8080
```

Then open `http://localhost:8080` and test the demo questions in `DEMO_SCRIPT.md`.
