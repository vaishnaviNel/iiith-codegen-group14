"""Code -> documentation with Qwen on the first N CoDocBench test samples (ROUGE-L).

Modes: zero_shot | rag (k=2 similar functions' docstrings) | rag_agent (rag + check-and-retry).
    python -m scripts.run_docs --n 100
Resumable: partial results per mode are saved after every sample.
"""

import argparse
import json
from pathlib import Path

from rouge_score import rouge_scorer

from src.agent.docs_agent import check_docstring, clean_docstring, run_doc_agent
from src.eval.code_eval import bootstrap_ci
from src.models.loader import ChatModel
from src.prompts import build_doc_messages
from src.rag.corpus import codoc_test
from src.rag.retriever import Retriever

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--k", type=int, default=2)
ap.add_argument("--out", default="results/qwen_docs.json")
args = ap.parse_args()

samples = codoc_test(args.n)
scorer = rouge_scorer.RougeScorer(["rougeL"])
model = ChatModel()
retriever = Retriever.load("indexes/codoc")
MAX_TOK = 220


def rouge(pred, ref):
    return scorer.score(ref, pred)["rougeL"].fmeasure


report = {}
cache = {}  # first RAG output per sample, reused as the agent's attempt 1
for mode in ["zero_shot", "rag", "rag_agent"]:
    part = Path(args.out).with_suffix(f".{mode}.partial.json")
    rows = json.loads(part.read_text(encoding="utf-8")) if part.exists() else []
    for i, s in enumerate(samples):
        if i < len(rows):
            continue
        examples = retriever.search(s["query"], k=args.k) if mode != "zero_shot" else []
        msgs = build_doc_messages(s["payload"]["code"], examples)
        if mode == "rag_agent":
            first = next(r for r in report["rag"]["rows"] if r["i"] == i)["raw"]
            r = run_doc_agent(msgs, s["payload"]["code"], lambda m: model.chat(m, max_new_tokens=MAX_TOK),
                              first_output=first)
            doc, extra = r["doc"], {"attempts": r["attempts"], "first_ok": r["first_ok"], "ok": r["ok"]}
            raw = doc
        else:
            raw = model.chat(msgs, max_new_tokens=MAX_TOK)
            doc = clean_docstring(raw)
            extra = {"ok": check_docstring(doc, s["payload"]["code"])[0]}
        rows.append({"i": i, "id": s["id"], "raw": raw, "doc": doc,
                     "rougeL": rouge(doc, s["payload"]["documentation"]), **extra})
        part.write_text(json.dumps(rows), encoding="utf-8")
        print(f"[{mode}] {i+1}/{len(samples)} rougeL={rows[-1]['rougeL']:.3f}", flush=True)
    vals = [r["rougeL"] for r in rows]
    lo, hi = bootstrap_ci(vals)
    report[mode] = {"rouge_l": sum(vals) / len(vals), "ci95": [lo, hi], "n": len(vals),
                    "check_pass_rate": sum(r["ok"] for r in rows) / len(rows), "rows": rows}
    print(f"== {mode}: ROUGE-L {report[mode]['rouge_l']:.4f}  CI {lo:.4f}-{hi:.4f}  "
          f"docstring-check pass {report[mode]['check_pass_rate']:.0%}", flush=True)

Path(args.out).write_text(json.dumps(report), encoding="utf-8")
