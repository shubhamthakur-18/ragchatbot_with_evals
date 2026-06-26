# 🤖 Evaluation-Driven RAG Chatbot

> This project is a research-style **Retrieval-Augmented Generation (RAG)** chatbot built as a portfolio project with **integrated evaluation**. Instead of stopping at a working chatbot, the project **tracks four RAG versions, evaluates each one with RAGAs, and uses the results to guide retrieval and generation improvements**.

The chatbot answers questions from retrieved context only. The current version uses a hybrid retriever that combines dense semantic search with BM25 keyword search, then blends both rankings with Reciprocal Rank Fusion (RRF).

## Why This Project Exists

RAG systems are easy to demo and hard to trust. A chatbot can sound fluent even when it retrieves weak context, ignores evidence, or produces unsupported answers. This project treats RAG quality as an engineering problem:

- Build a baseline RAG pipeline.
- Measure it with repeatable evals.
- Change one part of the system at a time.
- Compare faithfulness, relevance, precision, and recall.
- Keep JSON and HTML reports for inspection.

The goal is to show the full loop: ingestion, retrieval, generation, evaluation, diagnosis, and iteration.

## Tech Stack

- Python
- ChromaDB for persistent vector storage
- Sentence Transformers: `all-MiniLM-L6-v2`
- LiteLLM for model access
- RAGAs for automated RAG evaluation
- RAGBench `delucionqa` dataset for documents and test samples
- BM25 via `rank-bm25`
- Reciprocal Rank Fusion for hybrid retrieval

## Project Structure

```text
.rag-with-evals
|
├── v4/
|   ├──ingest.py               # Loads, deduplicates, chunks, embeds, and stores documents
|   ├──chatbot.py              # Current RAG chatbot implementation
|   ├──eval.py                 # RAGAs evaluation pipeline and HTML report generator
|   ├── eval_results.json      # Latest evaluation results
|   └── eval_report.html       # Latest visual evaluation report
├── v1/                        # Dense retrieval baseline
├── v2/                        # Hybrid retrieval experiment
├── v3/                        # Follow-up experiment and evaluation snapshot
├── chroma_db/                 # Local ChromaDB vector store
└── requirements.txt           # Requirements file with dependencies library
```

## Research Method

Each version is treated as an experiment with a hypothesis, implementation change, and measured result.

| Version | Main Change | Research Hypothesis | Result |
| --- | --- | --- | --- |
| V1 | Dense vector retrieval only, top-5 chunks | Semantic similarity is enough for grounded QA over the indexed corpus. | Strong faithfulness, but weaker answer relevancy and context recall. |
| V2 | Added BM25 keyword retrieval and RRF rank fusion | Combining lexical and semantic retrieval should recover more exact-match evidence. | Answer relevancy improved from `0.5661` to `0.6653`; context recall improved from `0.7222` to `0.7857`. |
| V3 | Continued hybrid retrieval evaluation | Re-running the hybrid approach helps expose stability and metric tradeoffs. | The run showed a regression, especially in precision and relevance, making the need for controlled evals visible. |
| V4 / Current | Hybrid retrieval with larger context budget and deterministic generation | More retrieved evidence plus temperature `0` should improve groundedness and reduce answer variance. | Faithfulness recovered to `0.9571`, context recall reached `0.7833`, and final average score is the strongest of the saved runs. |

## Evaluation Results

All saved versions were evaluated on 15 samples from RAGBench `delucionqa` using RAGAs.

| Version | Faithfulness | Answer Relevancy | Context Precision | Context Recall | Average |
| --- | ---: | ---: | ---: | ---: | ---: |
| V1 | 1.0000 | 0.5661 | 0.6759 | 0.7222 | 0.7411 |
| V2 | 0.9000 | 0.6653 | 0.6444 | 0.7857 | 0.7489 |
| V3 | 0.8833 | 0.5877 | 0.6111 | 0.7500 | 0.7080 |
| V4 / Current | 0.9571 | 0.6529 | 0.6076 | 0.7833 | 0.7502 |

Key takeaways:

- Dense retrieval was highly faithful but missed some useful context.
- Hybrid retrieval improved recall by combining semantic and lexical signals.
- RRF helped avoid relying too heavily on a single retrieval method.
- The latest version has the best overall saved score while preserving high faithfulness.
- The non-monotonic V3 result is useful: it shows why integrated evals matter in RAG development.

## Current RAG Pipeline

1. `ingest.py` loads documents from RAGBench `delucionqa`.
2. Documents are deduplicated to reduce repeated evidence.
3. Text is split into overlapping chunks with `RecursiveCharacterTextSplitter`.
4. Chunks are embedded with `all-MiniLM-L6-v2`.
5. Embeddings and documents are stored in ChromaDB.
6. At query time, the chatbot retrieves candidates using dense vector search from ChromaDB and BM25 keyword search over stored chunks.
7. Results are fused with Reciprocal Rank Fusion.
8. The top context chunks are passed to the LLM.
9. The system prompt instructs the model to answer only from provided context.

## Integrated Evals

`eval.py` runs the complete RAG pipeline and evaluates generated answers with RAGAs:

- `faithfulness`: whether the answer is supported by retrieved context
- `answer_relevancy`: whether the answer addresses the question
- `context_precision`: whether retrieved chunks are useful and focused
- `context_recall`: whether retrieved chunks contain the needed evidence

The evaluator writes:

- `eval_results.json`: aggregate and per-sample metrics
- `eval_report.html`: visual dashboard with score cards, radar chart, per-sample answers, ground truth, and retrieved contexts

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file with your model configuration:

```env
LITELLM_MODEL=your-model-name
OPENROUTER_API_KEY=your-openrouter-key
# or
OPENAI_API_KEY=your-openai-key
```

## Run The Project

Build the vector store:

```bash
python ingest.py
```

Start the chatbot:

```bash
python chatbot.py
```

Run evaluation:

```bash
python eval.py
```

Open `eval_report.html` in a browser to inspect aggregate and per-sample behavior.

## Portfolio Highlights

This project demonstrates:

- Practical RAG architecture design
- Dense retrieval and hybrid retrieval implementation
- Evaluation-driven development with RAGAs
- Use of benchmark data instead of hand-picked examples
- Retrieval diagnostics through precision and recall metrics
- Grounded generation through strict context-only prompting
- Experiment tracking across multiple chatbot versions
- Production-relevant reporting via JSON and HTML outputs

## Future Improvements

- Add query rewriting for ambiguous or underspecified questions.
- Add reranking after RRF to improve context precision.
- Expand the eval sample size beyond 15 examples.
- Track cost, latency, and token usage per version.
- Add CI-style regression checks for RAG quality gates.
- Compare multiple embedding models and chunking strategies.

