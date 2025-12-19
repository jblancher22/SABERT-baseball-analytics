# corpus_builder.py

import os
from typing import List

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# --- Configuration Constants ---
EMBED_MODEL_NAME='sentence-transformers/bert-base-nli-mean-tokens'
CHROMA_DIR = ".chroma_baseball"
DOCS_FOLDER = "Docs"
CHUNK_SIZE = 1500
OVERLAP = 200


# -------------------------------

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> List[str]:
    """Split a long string into overlapping chunks."""
    chunks = []
    start = 0
    n = len(text)
    if n == 0:
        return chunks
    while start < n:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def build_corpus(folder: str, embed_model: SentenceTransformer, corpus_collection: chromadb.Collection) -> None:
    """
    Scan a folder for .txt files, read them, chunk them, embed them,
    and add them to the 'baseball_corpus' collection.
    """
    docs: List[str] = []
    sources: List[str] = []

    # Check if the folder exists
    if not os.path.isdir(folder):
        print(f"Error: Docs folder '{folder}' not found. Please create it and add .txt files.")
        return

    for filename in os.listdir(folder):
        if not filename.endswith(".txt"):
            continue

        path = os.path.join(folder, filename)
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read().strip()

        if not text:
            continue

        chunks = chunk_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP)

        for idx, chunk in enumerate(chunks):
            docs.append(chunk)
            sources.append(f"{filename}#chunk_{idx}")

    print(f"Embedding {len(docs)} chunks...")
    embeddings = embed_model.encode(docs, convert_to_numpy=False)

    print("Adding chunks to ChromaDB...")
    # Batch addition for efficiency
    ids = [f"doc_{i}" for i in range(len(docs))]
    metadatas = [{"source": src} for src in sources]
    embeddings_list = [emb.tolist() for emb in embeddings]

    corpus_collection.add(
        ids=ids,
        documents=docs,
        metadatas=metadatas,
        embeddings=embeddings_list,
    )

    print(f"✅ Successfully indexed {len(docs)} chunks into baseball_corpus at {CHROMA_DIR}.")

def get_indexed_files(corpus_collection: chromadb.Collection) -> set:
    """
    Returns a set of filenames that have already been indexed in ChromaDB.
    """
    indexed_files = set()

    # Query everything in small batches (Chroma has no "get all" API)
    # n_results must be large enough to cover your dataset
    results = corpus_collection.get(include=["metadatas"])

    if results and "metadatas" in results:
        for meta in results["metadatas"]:
            source = meta.get("source", "")
            if "#" in source:
                filename = source.split("#")[0]
                indexed_files.add(filename)

    return indexed_files

def add_new_documents(folder: str, embed_model: SentenceTransformer, corpus_collection: chromadb.Collection):
    """
    Only processes documents that have NOT yet been indexed.
    """

    # 1. Find which files are already indexed
    indexed_files = get_indexed_files(corpus_collection)

    # 2. Loop through Docs folder
    for filename in os.listdir(folder):
        if not filename.endswith(".txt"):
            continue
        if filename in indexed_files:
            print(f"Skipping already-indexed file: {filename}")
            continue

        print(f"Indexing NEW file: {filename}")
        path = os.path.join(folder, filename)
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read().strip()

        if not text:
            continue

        chunks = chunk_text(text)

        docs = []
        sources = []
        ids = []
        for idx, chunk in enumerate(chunks):
            docs.append(chunk)
            sources.append(f"{filename}#chunk_{idx}")
            ids.append(f"{filename}_chunk_{idx}")

        embeddings = embed_model.encode(docs, convert_to_numpy=False)

        corpus_collection.add(
            ids=ids,
            documents=docs,
            metadatas=[{"source": src} for src in sources],
            embeddings=[emb.tolist() for emb in embeddings],
        )

        print(f"✔️ Added {len(chunks)} new chunks from {filename}")


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    print(f"Loading embedding model: {EMBED_MODEL_NAME}")
    # Load the model outside the main class to keep the RAG runtime lean
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)

    print(f"Initializing persistent ChromaDB client at {CHROMA_DIR}")
    chroma_client = chromadb.PersistentClient(
        path=CHROMA_DIR,
        settings=Settings(anonymized_telemetry=False),
    )

    # Collection that holds the main baseball corpus
    corpus_collection = chroma_client.get_or_create_collection("baseball_corpus")

    if corpus_collection.count() == 0:
        print("Corpus is empty. Building full corpus...")
        build_corpus(DOCS_FOLDER, embed_model, corpus_collection)
    else:
        print("Corpus already exists. Checking for new documents...")
        add_new_documents(DOCS_FOLDER, embed_model, corpus_collection)