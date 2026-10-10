from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .rag import answer_query, build_index

store = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global store
    store = build_index(force=os.getenv("REBUILD_INDEX", "0") == "1")
    yield

app = FastAPI(
    title="FinBase Financial Support RAG Assistant",
    version="1.0.0",
    description="Grounded customer-support assistant over the six provided FinBase policy PDFs.",
    lifespan=lifespan,
)

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    history: list[Message] = Field(default_factory=list)
    top_k: int = Field(default=5, ge=1, le=10)

@app.get("/health")
def health():
    return {"status": "healthy", "indexed_chunks": len(store.chunks) if store else 0}

@app.post("/api/chat")
def chat(req: ChatRequest):
    result = answer_query(req.query, store, [m.model_dump() for m in req.history], req.top_k)
    return result

@app.get("/", response_class=HTMLResponse)
def ui():
    return HTMLResponse(INDEX_HTML)

INDEX_HTML = r'''<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FinBase Support Assistant</title>
<style>
body{font-family:Inter,system-ui,sans-serif;background:#f4f7fb;margin:0;color:#172033}.wrap{max-width:1000px;margin:0 auto;padding:28px}.hero{background:#132238;color:white;border-radius:18px;padding:28px}.hero h1{margin:0 0 8px}.hero p{margin:0;opacity:.85}.chat{margin-top:18px;background:white;border-radius:18px;padding:20px;box-shadow:0 8px 30px #15243a12}.messages{min-height:360px;max-height:58vh;overflow:auto}.msg{margin:14px 0;padding:14px 16px;border-radius:14px;white-space:pre-wrap}.user{background:#e8f0ff;margin-left:15%}.bot{background:#f1f4f8;margin-right:8%}.sources{margin-top:10px;font-size:13px}.source{border-left:3px solid #5878a8;padding:8px 12px;margin:8px 0;background:#f8fafc}.score{float:right;color:#65748b}.row{display:flex;gap:10px;margin-top:14px}.row input{flex:1;padding:14px;border:1px solid #ccd5e0;border-radius:12px;font-size:15px}.row button{padding:0 22px;border:0;border-radius:12px;background:#132238;color:#fff;font-weight:700;cursor:pointer}.hint{color:#637084;font-size:13px;margin-top:8px}.status{font-size:12px;margin-top:8px;color:#526174}
</style></head><body><div class="wrap"><div class="hero"><h1>FinBase Financial Support Assistant</h1><p>Document-grounded RAG • cited sources • no-guess policy</p></div>
<div class="chat"><div id="messages" class="messages"><div class="msg bot">Hi! Ask me about FinBase savings accounts, payments, credit cards, personal loans, wealth products, or KYC/security policies. I will cite the document and page used.</div></div>
<div class="row"><input id="q" placeholder="e.g. What is the foreclosure charge after 26 months?"/><button onclick="send()">Ask</button></div><div class="hint">Tip: follow-up questions are supported using the recent conversation context.</div><div id="status" class="status"></div></div></div>
<script>
const messages=[]; const box=document.getElementById('messages'); const q=document.getElementById('q');
q.addEventListener('keydown',e=>{if(e.key==='Enter')send()});
function add(role,text,sources=[]){const d=document.createElement('div');d.className='msg '+role;d.textContent=text;if(sources.length){const s=document.createElement('div');s.className='sources';s.innerHTML='<b>Sources</b>'+sources.map(x=>`<div class="source"><b>${x.document}</b> — page ${x.page} <span class="score">score ${x.score}</span><br>${x.snippet}</div>`).join('');d.appendChild(s)}box.appendChild(d);box.scrollTop=box.scrollHeight}
async function send(){const text=q.value.trim();if(!text)return;q.value='';add('user',text);messages.push({role:'user',content:text});document.getElementById('status').textContent='Retrieving grounded context…';try{const r=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({query:text,history:messages.slice(-5)})});const data=await r.json();add('bot',data.answer,data.sources||[]);messages.push({role:'assistant',content:data.answer});document.getElementById('status').textContent=data.grounded?'Grounded in the provided knowledge base.':'Not found in the provided knowledge base.'}catch(e){add('bot','The service could not process the request. Please try again.');document.getElementById('status').textContent='Request failed.'}}
</script></body></html>'''
