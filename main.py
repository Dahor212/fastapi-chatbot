from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import chromadb
import json
import logging

# Nastavení logování
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicializace FastAPI
app = FastAPI()

# Cesta k embeddingům na GitHubu
GITHUB_EMBEDDINGS_URL = "https://raw.githubusercontent.com/Dahor212/fastapi-chatbot/refs/heads/main/data/embeddings.json"

# Inicializace ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="docs")

def load_embeddings():
    """ Načtení embeddingů z GitHubu. """
    logger.info("📥 Načítám embeddingy z GitHubu...")
    response = requests.get(GITHUB_EMBEDDINGS_URL)
    if response.status_code == 200:
        data = response.json()
        logger.info("✅ Embeddingy úspěšně načteny z GitHubu!")
        return data  # Vrací seznam, ne slovník!
    else:
        logger.error("❌ Nepodařilo se načíst embeddingy! Status code: %d", response.status_code)
        return None

embeddings_data = load_embeddings()

if embeddings_data:
    for entry in embeddings_data:
        doc_id = entry["id"]  # Získání ID dokumentu
        embedding = entry["embedding"]  # Získání embeddingu
        collection.add(ids=[doc_id], embeddings=[embedding], metadatas=[entry.get("metadata", {})])  
    logger.info("✅ Embeddingy úspěšně uloženy do ChromaDB!")
else:
    logger.error("❌ Chyba při načítání embeddingů, aplikace nemusí fungovat správně!")

class QueryRequest(BaseModel):
    query: str

def get_query_embedding(query: str):
    """ Najde embedding pro daný dotaz. """
    for entry in embeddings_data:
        if entry["id"] == query:  # Předpokládá, že query odpovídá nějakému dokumentu
            return entry["embedding"]
    return None

@app.post("/chat")
async def chat(request: QueryRequest):
    logger.info("🔍 Přijatý dotaz: %s", request.query)
    
    query_embedding = get_query_embedding(request.query)
    if not query_embedding:
        logger.warning("⚠️ Embedding dotazu nebyl nalezen!")
        return {"response": "Embedding dotazu nebyl nalezen."}
    
    results = collection.query(query_embeddings=[query_embedding], n_results=3)
    
    if not results["documents"]:
        logger.warning("⚠️ Odpověď nebyla nalezena v databázi!")
        return {"response": "Odpověď nebyla nalezena v databázi."}
    
    logger.info("📄 Nalezené dokumenty: %s", results["documents"])
    response_text = "\n".join(results["documents"][0])
    
    return {"response": response_text}
