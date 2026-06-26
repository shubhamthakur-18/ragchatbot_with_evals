import os
from dotenv import load_dotenv
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import chromadb
import litellm
litellm.drop_params = True

load_dotenv()

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "hr_policy"
EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K = 10
RRF_K = 60              # RRF constant - higher = smoother rank blending
FETCH_K = TOP_K * 3     # candidates fetched from each source before fusion
MODEL = os.getenv("LITELLM_MODEL", "gpt-5-nano")

SYSTEM_PROMPT = (
    "Your are a helpful assistant. "
    "Answer the questions directly using ONLY the context provided. "
    "If the context does not contain enough information, say so clearly. "
    "Be concise and accurate."
)

RAG_TEMPLATE = """\
Context: {context}
Question: {question}
Answer based solely on the context above:"""


def load_retriever():
    """Return the ChromaDB collection and embedding model."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(COLLECTION_NAME)

    # Pull all chunks once - needed for build BM25 
    stored = collection.get(include=["documents"])
    all_ids : list[str] = stored["ids"]
    all_docs: list[str] = stored["documents"]

    # BM25 expects a list of token lists
    tokenized_corpus = [doc.lower().split() for doc in all_docs]
    bm25=BM25Okapi(tokenized_corpus)

    id_to_text = dict(zip(all_ids, all_docs))

    embedder = SentenceTransformer(EMBED_MODEL)

    print(f"    BM25 index built over {len(all_ids)} chunks.")
    return collection, embedder, bm25, all_ids, id_to_text


def _rrf_score(ranks: list[int]) -> float:
    return sum(1.0/ (RRF_K + r) for r in ranks)


def retrieve(
        query: str, 
        collection, 
        embedder, 
        bm25: BM25Okapi,
        all_ids: list[str],
        id_to_text: dict[str, str],
        top_k: int = TOP_K,
    ) -> list[str]:
    """
    Hybrid retrieval: Dense vector search + BM25 keyword search,
    combined with Reciprocal Rank Fusion.
    """
    # ----Dense vector search----
    vector = embedder.encode([query])[0].tolist()
    vec_result = collection.query(
        query_embeddings = [vector],
        n_results = FETCH_K,
        include = ["documents", "metadatas", "distances"],
    )
    vec_ids : list[str] = vec_result["ids"][0]

    # ---BM25 keyword search---
    scores = bm25.get_scores(query.lower().split())
    top_bm25_indices = np.argsort(scores)[::-1][:FETCH_K]
    bm25_ids : list[str] = [all_ids[i] for i in top_bm25_indices]

    # ---Reciprocal Rank Fusion---
    # Each score contributes {1/RRF_K + rank} to the doc's overall score
    rrf: dict[str, list[int]] = {}

    for rank, doc_id in enumerate(vec_ids):
        rrf.setdefault(doc_id, []).append(rank)

    for rank, doc_id in enumerate(bm25_ids):
        rrf.setdefault(doc_id, []).append(rank)
    
    fused = sorted(rrf.items(), key= lambda kv: _rrf_score(kv[1]), reverse=True)

    # Returns top-k texts, falling back gracefully if an id somehow isn't created
    results = []
    for doc_id, _ in fused[:top_k]:
        text = id_to_text.get(doc_id)
        if text: 
            results.append(text)

    return results


def generate(question: str, context_chunks: list[str]) -> str:
    """Call LiteLLM with the retrieved context and return the answer."""
    context = "\n\n---\n\n".join(context_chunks)
    user_message = RAG_TEMPLATE.format(context = context, question = question)

    response = litellm.completion(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0
    )
    return response.choices[0].message.content 


def chat_loop(collection, embedder, bm25, all_ids, id_to_text) -> None:
    print(f"RAG Chatbot (model: {MODEL}, retrieval: hybrid bm25 + vector)")
    print("Type your question and press Enter. Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not question:
            continue
        if question.lower() in {"quit", "exit"}:
            print("Goodbye.")
            break

        chunks = retrieve(question, collection, embedder, bm25, all_ids, id_to_text)
        answer = generate(question, chunks)
        print(f"\nAssistant: {answer}\n")


if __name__ == "__main__":
    print("Loading retreiver...")
    collection, embedder, bm25, all_ids, id_to_text = load_retriever()
    chat_loop(collection, embedder, bm25, all_ids, id_to_text)
