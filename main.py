import os
import docx
import openai
import chromadb
import logging
import uvicorn
from fastapi import FastAPI
from typing import List

# Nastavení logování
logging.basicConfig(level=logging.INFO)

# Načtení API klíče z prostředí (nastavíme na Railway)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logging.error("❌ API klíč pro OpenAI není nastaven! Ukončuji aplikaci.")
    exit(1)

openai.api_key = OPENAI_API_KEY

# Inicializace FastAPI
app = FastAPI()

# Určení cesty pro ChromaDB (Railway používá "/app" jako pracovní adresář)
db_path = os.getenv("CHROMA_DB_PATH", "/app/chroma_db")
os.makedirs(db_path, exist_ok=True)
chroma_client = chromadb.PersistentClient(path=db_path)
collection = chroma_client.get_or_create_collection("documents2")


def split_docx_to_chunks(file_path: str, chunk_size: int = 500) -> List[str]:
    """Rozdělení textu ze souboru DOCX na menší části."""
    doc = docx.Document(file_path)
    text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


def add_documents_to_chroma(file_path: str):
    """Načte dokument, rozdělí ho na části a přidá do ChromaDB."""
    if not os.path.exists(file_path):
        logging.error(f"❌ Soubor {file_path} nebyl nalezen.")
        return

    # Přidání souboru pouze pokud ještě není v databázi
    existing_docs = collection.count()
    logging.info(f"📊 ChromaDB obsahuje {existing_docs} dokumentů před načítáním.")

    chunks = split_docx_to_chunks(file_path)
    if not chunks:
        logging.error("❌ Nebyly nalezeny žádné textové části.")
        return

    # Získání již existujících ID
    existing_ids = set(collection.get()["ids"])

    new_documents = []
    new_ids = []
    new_metadatas = []

    for i, chunk in enumerate(chunks):
        doc_id = f"doc_{i}"
        if doc_id not in existing_ids:
            new_documents.append(chunk)
            new_ids.append(doc_id)
            new_metadatas.append({"source": file_path})

    if new_documents:
        collection.add(
            documents=new_documents,
            metadatas=new_metadatas,
            ids=new_ids
        )
        logging.info(f"✅ Přidáno {len(new_documents)} nových částí textu do ChromaDB.")
    else:
        logging.info("⚠️ Všechny části dokumentu již byly uloženy. Přeskakuji načítání.")


@app.on_event("startup")
def load_documents():
    """Načte dokumenty do ChromaDB při startu aplikace."""
    file_path = os.getenv("DOCX_FILE_PATH", "/app/documents/csobvypisy.docx")
    add_documents_to_chroma(file_path)


@app.get("/chat")
def chat(query: str):
    """Zodpovídání dotazů na základě dokumentů v ChromaDB."""
    logging.info(f"🔍 Přijatý dotaz: {query}")

    try:
        results = collection.query(query_texts=[query], n_results=1)

        if not results.get("documents") or not results["documents"][0]:
            logging.info("❌ ChromaDB nevrátil žádné relevantní dokumenty!")
            return {"response": "Tato informace není v databázi."}

        context = results["documents"][0][0]
        logging.info(f"📝 Použitý kontext: {context}")

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Odpovídej pouze na základě poskytnutých dokumentů."},
                {"role": "user", "content": f"Dotaz: {query}\n\nRelevantní informace: {context}"}
            ]
        )

        return {"response": response["choices"][0]["message"]["content"]}

    except Exception as e:
        logging.error(f"🚨 Chyba v /chat: {str(e)}")
        return {"error": "Interní chyba serveru, podívejte se do logu."}


@app.get("/debug")
def debug():
    """Vrátí počet dokumentů v ChromaDB a ukázku několika z nich."""
    documents = collection.get()
    if "ids" not in documents or len(documents["ids"]) == 0:
        logging.warning("⚠️ ChromaDB je prázdná!")
        return {"count": 0, "sample": []}

    count = len(documents["ids"])
    sample = [{"id": doc_id, "text": text} for doc_id, text in zip(documents["ids"][:5], documents["documents"][:5])]
    logging.info(f"📊 ChromaDB obsahuje {count} dokumentů.")

    return {"count": count, "sample": sample}


# Spuštění serveru při lokálním testování
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
