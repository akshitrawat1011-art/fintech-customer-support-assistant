from __future__ import annotations
import json
import re
from pathlib import Path
from app.rag import build_index, answer_query

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "evaluation" / "eval_set.json"

def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()

def main():
    store = build_index()
    cases = json.loads(DATA.read_text(encoding="utf-8"))
    retrieval_hits = 0
    reciprocal = []
    answer_hits = 0
    citation_hits = 0
    grounded_hits = 0
    not_found_hits = 0
    for case in cases:
        results = store.search(case["question"], 5)
        expected = case["source"]
        rank = next((i+1 for i,r in enumerate(results) if r["chunk"].source == expected), None)
        retrieval_hits += rank is not None
        reciprocal.append(1/rank if rank else 0)
        out = answer_query(case["question"], store)
        answer_hits += int(any(norm(x) in norm(out["answer"]) for x in case.get("answer_contains", [])))
        if not case.get("not_found"):
            citation_hits += int(any(s["document"] == expected and (not case.get("page") or s["page"] == case["page"]) for s in out["sources"]))
            grounded_hits += int(out["grounded"] and bool(out["sources"]))
        nf = case.get("not_found", False)
        if nf:
            not_found_hits += int(not out["grounded"])
    answerable=[c for c in cases if not c.get("not_found")]
    n=len(cases)
    a=len(answerable)
    metrics={
        "cases": n,
        "answerable_cases": a,
        "retrieval_hit_at_5": round(retrieval_hits/a,3),
        "mrr_at_5": round(sum(reciprocal)/a,3),
        "answer_keyword_accuracy": round(answer_hits/a,3),
        "citation_accuracy": round(citation_hits/a,3),
        "grounded_response_rate": round(grounded_hits/a,3),
        "not_found_detection": round(not_found_hits/max(1,sum(1 for c in cases if c.get("not_found"))),3),
    }
    print(json.dumps(metrics, indent=2))
    (ROOT/"evaluation"/"results.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")

if __name__ == "__main__": main()
