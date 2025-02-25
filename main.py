import json
import os
import requests
import chromadb
import uvicorn
from fastapi import FastAPI, HTTPException, Request
import openai  # Přidáme OpenAI import
from starlette.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# Načtení API klíče z prostředí
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    print("❌ Chyba: OPENAI_API_KEY není nastaven!")
else:
    print("✅ OPENAI_API_KEY načten úspěšně.")

# URL k souboru s embeddingy na GitHubu
GITHUB_EMBEDDINGS_URL = "https://raw.githubusercontent.com/Dahor212/fastapi-chatbot/main/data/embeddings.json"

# Inicializace ChromaDB
try:
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection(name="documents")
    print("✅ ChromaDB inicializována úspěšně.")
except Exception as e:
    print(f"❌ Chyba při inicializaci ChromaDB: {e}")

# Načtení embeddingů z GitHubu s podrobným logováním
def load_embeddings_from_github():
    try:
        print(f"📥 Stahuji embeddingy z: {GITHUB_EMBEDDINGS_URL}")
        response = requests.get(GITHUB_EMBEDDINGS_URL, timeout=10)
        print(f"🔍 Status kód: {response.status_code}")

        if response.status_code != 200:
            print(f"❌ Chyba: GitHub vrátil status {response.status_code}")
            return None
        
        try:
            embeddings_data = response.json()
            if not isinstance(embeddings_data, dict):
                print("❌ Chyba: Načtená data nejsou validní JSON objekt.")
                return None

            print("✅ Embeddingy úspěšně načteny z GitHubu!")
            return embeddings_data

        except json.JSONDecodeError as e:
            print(f"❌ Chyba při dekódování JSON: {e}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"❌ Chyba při načítání embeddingů z GitHubu: {e}")
        return None

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("📥 Načítám embeddingy z GitHubu...")
    embeddings = load_embeddings_from_github()

    if embeddings:
        try:
            existing_ids = collection.get()["ids"]
            if existing_ids:
                collection.delete(ids=existing_ids)  # Správné mazání celé kolekce

            for doc_id, data in embeddings.items():
                if "embedding" in data:
                    collection.add(ids=[doc_id], embeddings=[data["embedding"]])

            print("✅ Embeddingy úspěšně uloženy do ChromaDB!")
        except Exception as e:
            print(f"❌ Chyba při ukládání do ChromaDB: {e}")

    yield
    print("🛑 Aplikace se ukončuje.")

app = FastAPI(lifespan=lifespan)

# Přidání CORS middleware
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
    return {"message": "Hello, world!"}

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
        # Kontrola, zda je API klíč OpenAI k dispozici
        if not openai.api_key:
            raise HTTPException(status_code=500, detail="OpenAI API key is missing")

        # Generování embeddingu pro dotaz
        query_embedding = openai.Embedding.create(input=query, model="text-embedding-ada-002")["data"][0]["embedding"]
    
    except openai.error.OpenAIError as e:
        print(f"❌ Chyba při připojení k OpenAI API: {e}")
        raise HTTPException(status_code=500, detail="Error connecting to OpenAI API")

    # Hledání v ChromaDB
    try:
        results = collection.query(query_embeddings=[query_embedding], n_results=1)

        if results["documents"]:
            return {"answer": results["documents"][0]}
        else:
            return JSONResponse(content={"message": "Answer not found in the database."}, status_code=404)
    
    except Exception as e:
        print(f"❌ Chyba při dotazu na ChromaDB: {e}")
        raise HTTPException(status_code=500, detail="Error querying ChromaDB")

if __name__ == "__main__":
    # Získej port z prostředí, nebo použij 8000 jako fallback
    port = int(os.getenv("PORT", 8000))
    print(f"🚀 Spouštím FastAPI na portu {port}")
    uvicorn.run("main:app", host="0.0.0.0", port=port)
