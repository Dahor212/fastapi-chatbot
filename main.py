import json
import os
import requests
import chromadb
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from starlette.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# URL k souboru s embeddingy na GitHubu (RAW verze!)
GITHUB_EMBEDDINGS_URL = "https://raw.githubusercontent.com/Dahor212/fastapi-chatbot/main/data/embeddings.json"

# Inicializace ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="documents")

# Funkce pro načtení embeddingů z GitHubu
def load_embeddings_from_github():
    try:
        response = requests.get(GITHUB_EMBEDDINGS_URL)
        response.raise_for_status()
        embeddings_data = response.json()
        print("✅ Embeddingy úspěšně načteny z GitHubu!")
        return embeddings_data
    except requests.exceptions.RequestException as e:
        print(f"❌ Chyba při načítání embeddingů z GitHubu: {e}")
        return None

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("📥 Načítám embeddingy z GitHubu...")
    embeddings = load_embeddings_from_github()

    if embeddings:
        existing_ids = collection.get()["ids"]
        if existing_ids:
            collection.delete(ids=existing_ids)
        for doc_id, data in embeddings.items():
            if "embedding" in data:
                collection.add(ids=[doc_id], embeddings=[data["embedding"]], documents=[data["text"]])
        print("✅ Embeddingy úspěšně uloženy do ChromaDB!")

    yield
    print("🛑 Aplikace se ukončuje.")

app = FastAPI(lifespan=lifespan)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Základní route pro /
@app.get("/")
async def root():
    return {"message": "API je spuštěno!"}

# Route pro favicon.ico
@app.get("/favicon.ico")
async def favicon():
    return JSONResponse(content={}, status_code=204)

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    query = data.get("query")

    if not query:
        raise HTTPException(status_code=400, detail="Query is required")

    try:
        # Načtení embeddingu dotazu z GitHubu
        embeddings = load_embeddings_from_github()
        if query in embeddings:
            query_embedding = embeddings[query]["embedding"]
        else:
            return JSONResponse(content={"message": "Embedding dotazu nebyl nalezen."}, status_code=404)
    except Exception as e:
        print(f"❌ Chyba při načítání embeddingu dotazu: {e}")
        raise HTTPException(status_code=500, detail="Error loading query embedding")

    # Hledání v ChromaDB
    results = collection.query(query_embeddings=[query_embedding], n_results=1)

    if results and "documents" in results and results["documents"]:
        return JSONResponse(content={"response": results["documents"][0]})
    else:
        return JSONResponse(content={"message": "Odpověď nebyla nalezena v databázi."}, status_code=404)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
