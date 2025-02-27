import os
import uvicorn
from fastapi import FastAPI, HTTPException
import requests
import chromadb
import json
from io import BytesIO
from docx import Document

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
    Tato funkce se pokouší najít embedding pro zadaný dotaz v obsahu (ne jen názvu souboru).
    """
    for entry in embeddings_data:
        if "metadata" in entry and "embedding" in entry:
            print(f"Checking embedding for query: {entry['metadata'].get('source')}")
            # Hledání textu, který odpovídá dotazu
            if query.lower() in entry["metadata"].get("source", "").lower():
                return entry["embedding"]
            # Pokud je to text dokumentu, vyhledej obsah textu pro odpověď
            if query.lower() in str(entry["embedding"]).lower():  # Porovnání s embeddingem
                return entry["embedding"]
    print("No matching embedding found for the query.")
    return None

def extract_text_from_docx(doc_url: str) -> str:
    """
    Načte obsah Word dokumentu (.docx) z URL a vrátí text.
    """
    response = requests.get(doc_url)
    if response.status_code == 200:
        doc = Document(BytesIO(response.content))
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        return '\n'.join(full_text)
    else:
        raise Exception("Nepodařilo se stáhnout soubor.")

@app.options("/chat")
def options_chat():
    return {}

@app.post("/chat")
def chat(request: dict):
    query = request.get("query")
    if not query:
        raise HTTPException(status_code=400, detail="Chybí dotaz.")
    
    query_embedding = get_query_embedding(query)
    if not query_embedding:
        return {"response": "Na tuto otázku nemám odpověď."}
    
    results = collection.query(query_embeddings=[query_embedding], n_results=1)
    print(f"Query results: {results}")  
    
    if results["ids"]:
        if results["metadatas"] and results["metadatas"][0]:
            metadata = results["metadatas"][0]
            doc_name = metadata[0].get('source', '')
            doc_url = f"https://raw.githubusercontent.com/Dahor212/fastapi-chatbot/refs/heads/main/soubory/{doc_name}"
            try:
                document_text = extract_text_from_docx(doc_url)
                return {"response": document_text}
            except Exception as e:
                return {"response": f"Chyba při načítání dokumentu: {str(e)}"}
            return {"response": "Na tuto otázku nemám odpověď."}
        else:
            return {"response": "Na tuto otázku nemám odpověď."}
    else:
        return {"response": "Na tuto otázku nemám odpověď."}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
