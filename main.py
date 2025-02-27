from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import requests
import chromadb
import json
import os

app = FastAPI()

# Povolení CORS pro frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# URL GitHub raw souboru s embeddingy
github_url = "https://raw.githubusercontent.com/Dahor212/fastapi-chatbot/main/data/embeddings.json"

# Načtení embeddingů z GitHubu
try:
    response = requests.get(github_url)
    response.raise_for_status()
    embeddings_data = response.json()
    print("✅ Embeddingy úspěšně načteny z GitHubu!")
except Exception as e:
    print(f"❌ Chyba při načítání embeddingů: {e}")
    embeddings_data = []

# Inicializace ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="documents")

# Uložení embeddingů do ChromaDB
for entry in embeddings_data:
    if "id" in entry and "embedding" in entry and "metadata" in entry:
        print(f"Adding embedding with ID: {entry['id']}")  # Logování přidání embeddingu
        collection.add(
            ids=[str(entry["id"])],
            embeddings=[entry["embedding"]],
            metadatas=[entry["metadata"]]
        )
print("✅ Embeddingy úspěšně uloženy do ChromaDB!")

def get_query_embedding(query: str):
    """
    Tato funkce se pokouší najít embedding pro zadaný dotaz v metadatech.
    """
    for entry in embeddings_data:
        if "metadata" in entry:
            print(f"Checking embedding for query: {entry['metadata'].get('source')}")
            if entry["metadata"].get("source") == query:
                return entry["embedding"]
    print("No matching embedding found for the query.")
    return None

@app.options("/chat")
def options_chat():
    """
    Odpověď na OPTIONS požadavek pro CORS.
    """
    return {}

@app.post("/chat")
def chat(request: dict):
    """
    Hlavní funkce pro zpracování dotazu, která se pokusí získat odpověď z ChromaDB.
    """
    query = request.get("query")
    if not query:
        raise HTTPException(status_code=400, detail="Chybí dotaz.")
    
    # Získání embeddingu pro dotaz
    query_embedding = get_query_embedding(query)
    if not query_embedding:
        return {"response": "Na tuto otázku nemám odpověď."}
    
    # Dotaz na ChromaDB
    results = collection.query(query_embeddings=[query_embedding], n_results=1)
    print(f"Query results: {results}")  # Logování výsledků dotazu
    
    if results["ids"]:
        return {"response": results["metadatas"][0]["source"]}
    else:
        return {"response": "Na tuto otázku nemám odpověď."}

@app.get("/")
def root():
    """
    Základní endpoint pro testování, zda API běží.
    """
    return {"message": "API běží!"}
