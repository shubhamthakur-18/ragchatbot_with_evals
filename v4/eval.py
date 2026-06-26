"""
RAGAs evaluation for the RAG chatbot.

Loads test questions from ragbench/delucionqa, runs the full RAG pipeline
(retrieve + generate), evaluates with RAGAs metrics, and writes:
  - eval_results.json  — per-sample and aggregate scores
  - eval_report.html   — modern HTML dashboard

Usage:
  python eval.py

Prerequisites:
  - Run ingest.py first to populate ChromaDB
  - Ensure LITELLM_MODEL and OPENROUTER_API_KEY (or OPENAI_API_KEY) are set in .env
"""

import os
import sys
import json
import datetime
from dotenv import load_dotenv

# Ensure Unicode characters print correctly on Windows terminals.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()

# ── Configuration ────────────────────────────────────────────────────────────
N_SAMPLES       = 15
RAGBENCH_SUBSET = "delucionqa"
OUTPUT_JSON     = "eval_results.json"
OUTPUT_HTML     = "eval_report.html"
EMBED_MODEL     = "all-MiniLM-L6-v2"
# ─────────────────────────────────────────────────────────────────────────────

_METRIC_COLORS = {
    "faithfulness":      "#4ade80",
    "answer_relevancy":  "#60a5fa",
    "context_precision": "#f59e0b",
    "context_recall":    "#c084fc",
}


# ── Data loading ─────────────────────────────────────────────────────────────

def load_test_samples(n: int) -> list[dict]:
    """Pull N labelled samples (question + ground-truth answer) from ragbench."""
    from datasets import load_dataset

    print(f"Loading {n} samples from rungalileo/ragbench ({RAGBENCH_SUBSET})…")
    ds = None
    for split in ("test", "train"):
        try:
            ds = load_dataset("rungalileo/ragbench", RAGBENCH_SUBSET, split=split)
            print(f"  Using split='{split}'")
            break
        except Exception:
            continue

    if ds is None:
        raise RuntimeError("Could not load ragbench dataset.")

    samples = []
    for row in ds:
        if len(samples) >= n:
            break
        q  = (row.get("question") or "").strip()
        gt = (row.get("response") or row.get("answer") or "").strip()
        if q and gt:
            samples.append({"question": q, "ground_truth": gt})

    if not samples:
        raise RuntimeError(
            "No samples with both 'question' and 'answer' fields found in the dataset."
        )
    print(f"  Loaded {len(samples)} samples.\n")
    return samples


# ── RAG pipeline ─────────────────────────────────────────────────────────────

def run_rag_pipeline(samples: list[dict], collection, embedder, bm25, all_ids, id_to_text) -> list[dict]:
    """Retrieve context + generate answer for each sample."""
    from V4.chatbot import retrieve, generate

    records = []
    total = len(samples)
    for i, s in enumerate(samples, 1):
        q = s["question"]
        print(f"  [{i:>2}/{total}] {q[:70]}{'…' if len(q) > 70 else ''}")
        contexts = retrieve(q, collection, embedder, bm25, all_ids, id_to_text)
        answer   = generate(q, contexts)
        records.append({
            "question":     q,
            "answer":       answer,
            "contexts":     contexts,
            "ground_truth": s["ground_truth"],
        })
    return records


# ── RAGAs helpers ─────────────────────────────────────────────────────────────

def _build_ragas_llm():
    """Build a LangchainLLMWrapper configured from .env."""
    from ragas.llms import LangchainLLMWrapper
    from langchain_openai import ChatOpenAI

    model_env = os.getenv("LITELLM_MODEL", "")

    if os.getenv("OPENAI_API_KEY"):
        lc_llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)
    
    
    elif model_env.startswith("openrouter/"):
        raw_model = model_env[len("openrouter/"):]        # e.g. "openai/gpt-oss-120b:free"
        api_key   = os.getenv("OPENROUTER_API_KEY", os.getenv("OR_API_KEY", ""))
        
        temperature = 1 if "gpt-5" in raw_model else 0
        
        lc_llm = ChatOpenAI(
            model=raw_model,
            api_key=api_key or "sk-dummy",
            base_url="https://openrouter.ai/api/v1",
            temperature=temperature,
        )
    
    else:
        raise EnvironmentError(
            "No LLM configured for RAGAs.\n"
            "Set LITELLM_MODEL=openrouter/<model> + OPENROUTER_API_KEY,\n"
            "or set OPENAI_API_KEY in .env."
        )
    return LangchainLLMWrapper(lc_llm)


def _build_ragas_embeddings():
    """Local sentence-transformers embeddings — no extra API key needed."""
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_community.embeddings import HuggingFaceEmbeddings

    return LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=EMBED_MODEL))


# ── RAGAs evaluation ──────────────────────────────────────────────────────────

def evaluate_with_ragas(records: list[dict]) -> tuple[dict, list[dict]]:
    """
    Run RAGAs on the records.
    Returns (aggregate_scores dict, per_sample_scores list[dict]).
    """
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall

    print("Building RAGAs LLM and embeddings…")
    ragas_llm = _build_ragas_llm()
    ragas_emb = _build_ragas_embeddings()

    metrics = [
        Faithfulness(),
        AnswerRelevancy(),
        ContextPrecision(),
        ContextRecall(),
    ]

    dataset = EvaluationDataset(samples=[
        SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            retrieved_contexts=r["contexts"],
            reference=r["ground_truth"],
        )
        for r in records
    ])

    print("Running RAGAs evaluation (makes LLM calls per metric per sample)…")
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_emb,
    )

    df = result.to_pandas()
    skip_cols = {"user_input", "response", "retrieved_contexts", "reference"}
    metric_cols = [c for c in df.columns if c not in skip_cols]

    aggregate  = {col: float(df[col].mean()) for col in metric_cols}
    per_sample = df[metric_cols].to_dict(orient="records")

    return aggregate, per_sample


# ── Persistence ───────────────────────────────────────────────────────────────

def save_results(
    records: list[dict],
    aggregate: dict,
    per_sample: list[dict],
    path: str,
) -> None:
    output = {
        "timestamp":        datetime.datetime.utcnow().isoformat(),
        "n_samples":        len(records),
        "model":            os.getenv("LITELLM_MODEL", "unknown"),
        "aggregate_scores": aggregate,
        "samples": [
            {**r, "scores": {k: (None if v != v else v) for k, v in s.items()}}
            for r, s in zip(records, per_sample)
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"Results saved -> {path}")


# ── HTML report ───────────────────────────────────────────────────────────────

def _score_color(v) -> str:
    """Traffic-light colour for a score badge."""
    if v is None or (isinstance(v, float) and v != v):
        return "#6b7280"
    if v >= 0.75:
        return "#4ade80"
    if v >= 0.5:
        return "#f59e0b"
    return "#f87171"


def generate_html_report(results_path: str, html_path: str) -> None:
    with open(results_path, encoding="utf-8") as f:
        data = json.load(f)

    aggregate = data["aggregate_scores"]
    samples   = data["samples"]
    ts        = data["timestamp"][:19].replace("T", " ")
    model     = data.get("model", "unknown")

    # ── Score cards ──
    cards_html = ""
    for metric, score in aggregate.items():
        color = _METRIC_COLORS.get(metric, "#6b7280")
        pct   = round(score * 100)
        label = metric.replace("_", " ").title()
        cards_html += f"""
      <div class="card" style="--c:{color}">
        <div class="card-label">{label}</div>
        <div class="card-score">{score:.3f}</div>
        <div class="bar-track"><div class="bar-fill" style="width:{pct}%;background:{color}"></div></div>
        <div class="card-pct">{pct}%</div>
      </div>"""

    # ── Table rows ──
    metric_names = list(aggregate.keys())
    metric_headers = "".join(
        f'<th style="color:{_METRIC_COLORS.get(m,"#6b7280")}">'
        f'{m.replace("_"," ").title()}</th>'
        for m in metric_names
    )

    rows_html = ""
    for i, s in enumerate(samples):
        sc = s.get("scores", {})
        score_cells = ""
        for m in metric_names:
            val = sc.get(m)
            if val is not None and not (isinstance(val, float) and val != val):
                score_cells += (
                    f'<td><span class="badge" '
                    f'style="background:{_score_color(val)};color:#000">'
                    f'{val:.2f}</span></td>'
                )
            else:
                score_cells += "<td><span style='color:#6b7280'>—</span></td>"

        q   = s["question"]
        ans = s["answer"]
        gt  = s["ground_truth"]
        ctx_items = "".join(
            f'<li class="ctx-item">{c[:240]}{"…" if len(c) > 240 else ""}</li>'
            for c in s["contexts"]
        )
        rows_html += f"""
      <tr>
        <td class="num">{i + 1}</td>
        <td class="q-cell">{q}</td>
        <td class="ans-cell">{ans[:300]}{"…" if len(ans) > 300 else ""}</td>
        <td class="gt-cell">{gt[:240]}{"…" if len(gt) > 240 else ""}</td>
        {score_cells}
        <td>
          <button class="toggle" onclick="tog(this)">&#9654; Show</button>
          <ul class="ctx-list" hidden>{ctx_items}</ul>
        </td>
      </tr>"""

    # ── Radar chart data ──
    labels_js = json.dumps([m.replace("_", " ").title() for m in metric_names])
    values_js = json.dumps([round(aggregate[m], 4) for m in metric_names])
    colors_js = json.dumps([_METRIC_COLORS.get(m, "#6b7280") for m in metric_names])

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>RAG Evaluation Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
:root{{
  --bg:#0d1117;--s1:#161b22;--s2:#21262d;
  --border:#30363d;--txt:#e6edf3;--muted:#8b949e;--accent:#7c3aed
}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:var(--bg);color:var(--txt);font-family:system-ui,sans-serif;line-height:1.6;min-height:100vh}}

/* Header */
header{{
  background:linear-gradient(135deg,#0d1117 0%,#13092e 50%,#0d1117 100%);
  border-bottom:1px solid var(--border);padding:2.75rem 1.5rem;text-align:center
}}
.header-title{{
  font-size:2rem;font-weight:800;letter-spacing:-1px;
  background:linear-gradient(90deg,#a78bfa 0%,#60a5fa 100%);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent
}}
.header-sub{{color:var(--muted);margin-top:.4rem;font-size:.9rem}}
.badge{{
  display:inline-block;background:rgba(124,58,237,.2);
  border:1px solid rgba(124,58,237,.45);color:#a78bfa;
  padding:.25rem .85rem;border-radius:9999px;font-size:.75rem;margin-top:.8rem
}}

main{{max-width:1320px;margin:0 auto;padding:2rem 1.5rem}}

.section{{margin-bottom:2.75rem}}
.sec-title{{
  font-size:.72rem;font-weight:700;letter-spacing:1.8px;
  text-transform:uppercase;color:var(--muted);
  margin-bottom:1rem;padding-bottom:.5rem;border-bottom:1px solid var(--border)
}}

/* Score cards */
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:1rem}}
.card{{
  background:var(--s1);border:1px solid var(--border);border-radius:14px;
  padding:1.4rem 1.25rem;text-align:center;position:relative;overflow:hidden;
  transition:transform .2s,box-shadow .2s
}}
.card:hover{{transform:translateY(-3px);box-shadow:0 16px 40px rgba(0,0,0,.5)}}
.card::before{{
  content:'';position:absolute;top:0;left:0;right:0;
  height:3px;background:var(--c);border-radius:14px 14px 0 0
}}
.card-label{{font-size:.68rem;text-transform:uppercase;letter-spacing:1.2px;color:var(--muted);margin-bottom:.4rem}}
.card-score{{font-size:2.6rem;font-weight:800;color:var(--c);line-height:1}}
.bar-track{{background:var(--s2);border-radius:4px;height:5px;overflow:hidden;margin:.75rem 0 .4rem}}
.bar-fill{{height:100%;border-radius:4px;transition:width 1.4s cubic-bezier(.22,1,.36,1)}}
.card-pct{{font-size:.8rem;color:var(--muted)}}

/* Chart panel */
.chart-panel{{
  background:var(--s1);border:1px solid var(--border);border-radius:14px;
  padding:1.75rem;display:flex;justify-content:center;align-items:center
}}
.chart-panel canvas{{max-width:440px;max-height:400px}}

/* Table */
.tbl-wrap{{background:var(--s1);border:1px solid var(--border);border-radius:14px;overflow:auto}}
table{{width:100%;border-collapse:collapse;font-size:.82rem}}
thead{{background:var(--s2)}}
th{{
  padding:.8rem 1rem;text-align:left;font-weight:700;
  font-size:.7rem;text-transform:uppercase;letter-spacing:.6px;
  color:var(--muted);border-bottom:1px solid var(--border);white-space:nowrap
}}
td{{padding:.8rem 1rem;border-bottom:1px solid var(--border);vertical-align:top}}
tr:last-child td{{border:none}}
tr:hover td{{background:rgba(255,255,255,.02)}}
.num{{color:var(--muted);font-size:.75rem;width:36px;text-align:center}}
.q-cell{{color:#a78bfa;max-width:200px;word-break:break-word}}
.ans-cell{{max-width:280px;word-break:break-word}}
.gt-cell{{color:#86efac;max-width:240px;word-break:break-word}}
.badge{{display:inline-block;padding:.18rem .55rem;border-radius:7px;font-size:.76rem;font-weight:700}}
.toggle{{
  background:rgba(124,58,237,.15);border:1px solid rgba(124,58,237,.4);
  color:#a78bfa;padding:.2rem .65rem;border-radius:7px;
  cursor:pointer;font-size:.72rem;transition:background .15s;white-space:nowrap
}}
.toggle:hover{{background:rgba(124,58,237,.35)}}
.ctx-list{{margin-top:.5rem;padding:0;list-style:none}}
.ctx-item{{
  background:var(--bg);border-left:2px solid rgba(124,58,237,.4);
  padding:.45rem .7rem;margin-bottom:.35rem;
  font-size:.78rem;color:var(--muted);border-radius:0 5px 5px 0;line-height:1.5
}}

footer{{
  text-align:center;padding:1.75rem;color:var(--muted);
  font-size:.8rem;border-top:1px solid var(--border);margin-top:1rem
}}
</style>
</head>
<body>

<header>
  <div class="header-title">RAG Evaluation Report</div>
  <div class="header-sub">RAGAs Framework &nbsp;·&nbsp; DelucionQA Dataset &nbsp;·&nbsp; {len(samples)} samples</div>
  <span class="badge">Generated {ts} UTC</span>
</header>

<main>

  <div class="section">
    <div class="sec-title">Aggregate Scores</div>
    <div class="cards">{cards_html}
    </div>
  </div>

  <div class="section">
    <div class="sec-title">Metrics Overview</div>
    <div class="chart-panel"><canvas id="rc"></canvas></div>
  </div>

  <div class="section">
    <div class="sec-title">Per-Sample Results ({len(samples)} samples)</div>
    <div class="tbl-wrap">
      <table>
        <thead>
          <tr>
            <th>#</th>
            <th>Question</th>
            <th>Generated Answer</th>
            <th>Ground Truth</th>
            {metric_headers}
            <th>Contexts</th>
          </tr>
        </thead>
        <tbody>{rows_html}
        </tbody>
      </table>
    </div>
  </div>

</main>

<footer>
  Powered by <strong>RAGAs</strong> &nbsp;·&nbsp; Model: <code>{model}</code>
</footer>

<script>
function tog(btn) {{
  var ul = btn.nextElementSibling;
  var show = ul.hidden;
  ul.hidden = !show;
  btn.innerHTML = show ? '&#9660; Hide' : '&#9654; Show';
}}

new Chart(document.getElementById('rc'), {{
  type: 'radar',
  data: {{
    labels: {labels_js},
    datasets: [{{
      label: 'Score',
      data: {values_js},
      backgroundColor: 'rgba(124,58,237,.12)',
      borderColor: '#7c3aed',
      borderWidth: 2.5,
      pointBackgroundColor: {colors_js},
      pointBorderColor: '#0d1117',
      pointBorderWidth: 2,
      pointRadius: 6,
      pointHoverRadius: 8,
    }}]
  }},
  options: {{
    responsive: true,
    scales: {{
      r: {{
        min: 0,
        max: 1,
        ticks: {{
          stepSize: 0.25,
          color: '#8b949e',
          backdropColor: 'transparent',
          font: {{ size: 11 }}
        }},
        grid: {{ color: '#21262d' }},
        pointLabels: {{ color: '#c9d1d9', font: {{ size: 12, weight: '700' }} }},
        angleLines: {{ color: '#30363d' }},
      }}
    }},
    plugins: {{
      legend: {{ display: false }},
      tooltip: {{
        backgroundColor: '#161b22',
        borderColor: '#30363d',
        borderWidth: 1,
        callbacks: {{
          label: function(ctx) {{ return '  ' + ctx.parsed.r.toFixed(3); }}
        }}
      }}
    }}
  }}
}});
</script>
</body>
</html>"""

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"HTML report saved -> {html_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 50)
    print("  RAGAs Evaluation Framework")
    print("=" * 50)

    # 1. Load labelled test samples from ragbench
    samples = load_test_samples(N_SAMPLES)

    # 2. Load the RAG retriever (requires ingest.py to have been run)
    print("Loading RAG retriever (ChromaDB)…")
    from V4.chatbot import load_retriever
    collection, embedder, bm25, all_ids, id_to_text = load_retriever()

    # 3. Run the RAG pipeline on each sample
    print(f"\nRunning RAG pipeline on {len(samples)} samples…")
    records = run_rag_pipeline(samples, collection, embedder, bm25, all_ids, id_to_text)

    # 4. Evaluate with RAGAs
    print()
    aggregate, per_sample = evaluate_with_ragas(records)

    # 5. Persist results
    print()
    save_results(records, aggregate, per_sample, OUTPUT_JSON)

    # 6. Generate HTML report
    generate_html_report(OUTPUT_JSON, OUTPUT_HTML)

    # 7. Print summary
    print(f"\n{'=' * 50}")
    print("  Evaluation complete!")
    print(f"{'=' * 50}")
    for metric, score in aggregate.items():
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        print(f"  {metric:<25} {bar} {score:.4f}")
    print(f"\n  JSON  ->  {OUTPUT_JSON}")
    print(f"  HTML  ->  {OUTPUT_HTML}")
    print("=" * 50)


if __name__ == "__main__":
    main()

