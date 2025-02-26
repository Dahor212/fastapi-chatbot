from fastapi import FastAPI, HTTPException
import requests
import chromadb
import json
import os

app = FastAPI()

# URL GitHub raw souboru s embeddingy
github_url = "https://raw.githubusercontent.com/Dahor212/fastapi-chatbot/refs/heads/main/data/embeddings.json"

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
    collection.add(
        ids=[entry["id"]],
        embeddings=[entry["embedding"]],
        metadatas=[{"text": entry["text"]}]
    )
print("✅ Embeddingy úspěšně uloženy do ChromaDB!")

def get_query_embedding(query: str):
    for entry in embeddings_data:
        if entry["text"] == query:
            return entry["embedding"]
    return None

@app.post("/chat")
def chat(request: dict):
    query = request.get("query")
    if not query:
        raise HTTPException(status_code=400, detail="Chybí dotaz.")
    
    query_embedding = get_query_embedding(query)
    if not query_embedding:
        return {"response": "Na tuto otázku nemám odpověď."}
    
    results = collection.query(query_embeddings=[query_embedding], n_results=1)
    
    if results["ids"]:
        return {"response": results["metadatas"][0]["text"]}
    else:
        return {"response": "Na tuto otázku nemám odpověď."}

@app.get("/")
def root():
    return {"message": "API běží!"}
