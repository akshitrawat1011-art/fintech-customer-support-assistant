from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.rag import ingest_pdfs, LocalVectorStore, answer_query

def test_ingestion_has_all_documents():
    chunks=ingest_pdfs()
    names={c.source for c in chunks}
    assert names == {f"sample_{i}.pdf" for i in range(1,7)}
    assert len(chunks) > 500

def test_personal_loan_retrieval_and_citation():
    store=LocalVectorStore.build(ingest_pdfs())
    out=answer_query("What is the foreclosure charge after 26 months?",store)
    assert out["grounded"]
    assert any(s["document"]=="sample_5.pdf" for s in out["sources"])
    assert "1.5%" in out["answer"]

def test_not_found():
    store=LocalVectorStore.build(ingest_pdfs())
    out=answer_query("What is the weather forecast for Delhi tomorrow?",store)
    assert not out["grounded"]
    assert out["sources"] == []
