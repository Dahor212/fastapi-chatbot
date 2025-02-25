import json
import os
import requests
import chromadb
import uvicorn
from fastapi import FastAPI, HTTPException, Request
import openai
from starlette.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# Načtení API klíče z prostředí
openai.api_key = os.getenv("OPENAI_API_KEY")
if openai.api_key:
    print("✅ OPENAI_API_KEY načten úspěšně.")
else:
    print("❌ Chyba: OPENAI_API_KEY není nastaven!")

# URL k souboru s embeddingy na GitHubu
GITHUB_EMBEDDINGS_URL = "https://raw.githubusercontent.com/Dahor212/fastapi-chatbot/main/data/embeddings.json"

# Inicializace Chroma
try:
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection(name="documents")
    print("✅ ChromaDB inicializována úspěšně.")
except Exception as e:
    print(f"❌ Chyba při inicializaci ChromaDB: {e}")

# Načtení embeddingů z GitHubu
def load_embeddings_from_github():
    try:
        print(f"📥 Stahuji embeddingy z: {GITHUB_EMBEDDINGS_URL}")
        response = requests.get(GITHUB_EMBEDDINGS_URL)
        print(f"🔍 Status kód: {response.status_code}")
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
                collection.add(ids=[doc_id], embeddings=[data["embedding"]])
        print("✅ Embeddingy úspěšně uloženy do ChromaDB!")

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

@app.get("/")
async def root():
    return {"message": "Hello, world!"}

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
        print("🔍 Testuji připojení k OpenAI API...")
        response = requests.get("https://api.openai.com/v1/engines")
        print(f"🌐 Stavový kód: {response.status_code}")
        
        if response.status_code != 200:
            print(f"❌ OpenAI API nedostupné: {response.text}")
            raise HTTPException(status_code=500, detail="Cannot reach OpenAI API")

        print("🔍 Odesílám dotaz na OpenAI API...")
        query_embedding = openai.Embedding.create(input=query, model="text-embedding-ada-002")["data"][0]["embedding"]
        print("✅ Embedding pro dotaz úspěšně získán!")
    except openai.error.OpenAIError as e:
        print(f"❌ Chyba při připojení k OpenAI API: {e}")
        raise HTTPException(status_code=500, detail="Error connecting to OpenAI API")
    except requests.exceptions.RequestException as e:
        print(f"❌ Chyba při síťovém požadavku: {e}")
        raise HTTPException(status_code=500, detail="Network error while connecting to OpenAI API")
    except Exception as e:
        print(f"❌ Neočekávaná chyba: {e}")
        raise HTTPException(status_code=500, detail="Unexpected error")

    results = collection.query(query_embeddings=[query_embedding], n_results=1)

    if results["documents"]:
        return {"answer": results["documents"][0]}
    else:
        return JSONResponse(content={"message": "Answer not found in the database."}, status_code=404)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
