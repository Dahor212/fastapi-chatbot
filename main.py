import json
import os
import requests
import chromadb
import uvicorn
from fastapi import FastAPI
from openai import OpenAI
from starlette.responses import JSONResponse
from contextlib import asynccontextmanager

# URL k souboru s embeddingy na GitHubu (RAW verze!)
GITHUB_EMBEDDINGS_URL = "https://raw.githubusercontent.com/Dahor212/fastapi-chatbot/main/data/embeddings.json"

# Inicializace ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="documents")

# Načtení embeddingů z GitHubu
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

# Lifespan pro správnou inicializaci
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("📥 Načítám embeddingy z GitHubu...")
    embeddings = load_embeddings_from_github()

    if embeddings:
        collection.clear()
        for doc_id, embedding in embeddings.items():
            collection.add(ids=[doc_id], embeddings=[embedding])
        print("✅ Embeddingy úspěšně uloženy do ChromaDB!")
    
    yield
    print("Aplikace se ukončuje.")

# Inicializace FastAPI
app = FastAPI(lifespan=lifespan)

@app.get("/chat")
def chat(query: str):
    query_embedding = OpenAI().embeddings.create(input=query, model="text-embedding-ada-002")["data"][0]["embedding"]

    # Hledání v ChromaDB
    results = collection.query(query_embeddings=[query_embedding], n_results=1)

    if results["documents"]:
        return {"odpověď": results["documents"][0]}
    else:
        return JSONResponse(content={"message": "Odpověď nebyla nalezena v databázi."}, status_code=404)

# Spuštění aplikace
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Railway nastaví proměnnou prostředí PORT
    uvicorn.run(app, host="0.0.0.0", port=port)
