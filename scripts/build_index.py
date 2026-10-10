import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.rag import build_index
store = build_index(force=True)
print(f'Indexed {len(store.chunks)} chunks from data/*.pdf')
