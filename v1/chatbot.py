import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb
import litellm
litellm.drop_params = True

load_dotenv()

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "hr_policy"
EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5
MODEL = os.getenv("LITELLM_MODEL", "gpt-5-nano")

SYSTEM_PROMPT = (
    "Your are a helpful assistant. "
    "Answer questions using ONLY the context provided. "
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
    embedder = SentenceTransformer(EMBED_MODEL)
    return collection, embedder


def retrieve(query: str, collection, embedder, top_k: int = TOP_K) -> list[str]:
    """Embed the query and return the top-k most similar document chunks."""
    vector = embedder.encode([query])[0].tolist()
    results = collection.query(
        query_embeddings = [vector],
        n_results = top_k,
        include = ["documents", "metadatas", "distances"],
    )
    return results["documents"][0]


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
        temperature=0.2
    )
    return response.choices[0].message.content


def chat_loop(collection, embedder) -> None:
    print(f"RAG Chatbot (model: {MODEL})")
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

        chunks, ids = retrieve(question, collection, embedder)
        answer = generate(question, chunks)
        # print(f"\nAssistant: {answer}\nReference chunk IDs: {ids}")


if __name__ == "__main__":
    print("Loading retreiver...")
    collection, embedder = load_retriever()
    chat_loop(collection, embedder)
