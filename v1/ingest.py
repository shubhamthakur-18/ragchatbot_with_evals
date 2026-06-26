import os
import hashlib
from datasets import load_dataset
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb


CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "hr_policy"
EMBED_MODEL = "all-MiniLM-L6-v2"
RAGBENCH_SUBSET = "delucionqa"
MAX_DOCS = 500

CHUNK_SIZE = 512
CHUNK_OVERLAP = 64


def load_documents() -> list[str]:
    """Return a list of de-duplicated list of raw document strings from ragbench."""
    print(f"Loading rungalileo/ragbench ({RAGBENCH_SUBSET})...")
    ds = load_dataset("rungalileo/ragbench", RAGBENCH_SUBSET, split="train")

    seen: set[str] = set()
    docs: list[str] = []
    for row in ds: 
        for doc in row.get("documents", []):
            text = doc.strip()
            if not text: 
                continue
            key = hashlib.md5(text.encode()).hexdigest()
            if key not in seen:
                seen.add(key)
                docs.append(text)
                if len(docs) >= MAX_DOCS:
                    return docs
    return docs


def chunk_documents(docs: list[str]) -> list[dict]:
    """Split each document into overlapping chunks"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = CHUNK_SIZE, 
        chunk_overlap = CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for doc_idx, doc in enumerate(docs):
        for chunk_idx, chunk in enumerate(splitter.split_text(doc)):
            chunk_id = f"doc{doc_idx}_chunk{chunk_idx}"
            chunks.append({"id": chunk_id, "text": chunk})
    print(f"    {len(docs)} documents converted to {len(chunks)} chunks")
    return chunks


def embed_and_store(chunks: list[dict]) -> None:
    """Embed chunks with Sentence-transformers and persist to Chrome DB."""
    print(f"Loading embedding model '{EMBED_MODEL}'....")
    embedder = SentenceTransformer(EMBED_MODEL)

    print("Embedding chunks.....")
    texts = [c["text"] for c in chunks]
    vectors = embedder.encode(texts, batch_size=64, show_progress_bar=True)

    print(f"Writing to Chroma DB at '{CHROMA_PATH}'")
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Drop existing collections to re-runs start fresh
    try: 
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    batch_size = 512
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start: start + batch_size]
        collection.add(
            ids=[c["id"] for c in batch],
            documents=[c["text"] for c in batch],
            embeddings=vectors[start: start + batch_size].tolist(),
        )

    print(f"Done. {collection.count()} chunks stored in collection '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    docs = load_documents()
    chunks = chunk_documents(docs)
    embed_and_store(chunks)