"""RAG ablation on the MBPP VALIDATION split (never the test split — we tune here, report on test).

    python -m scripts.rag_ablation --n 40
Each variant's results are cached in results/ablation/<name>.json, so re-runs resume.
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset

from src.eval.code_eval import execute_candidate, wilson_interval
from src.models.loader import ChatModel
from src.prompts import build_code_messages, build_code_messages_reference
from src.rag.corpus import mbpp_corpus, mbpp_query
from src.rag.retriever import Retriever

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=40)
args = ap.parse_args()

# name: (index, query_uses_tests, k, min_score, prompt_style)
VARIANTS = {
    "zero_shot":        None,
    "turns_k3":         ("desc",  False, 3, 0.0,  "turns"),      # the version tested before
    "turns_k1_t80":     ("desc",  False, 1, 0.80, "turns"),      # fewer, only close neighbours
    "ref_k3":           ("desc",  False, 3, 0.0,  "reference"),  # context-with-warning prompt
    "ref_k2_tests":     ("tests", True,  2, 0.0,  "reference"),  # test-aware retrieval
    "ref_k2_tests_t75": ("tests", True,  2, 0.75, "reference"),  # + similarity floor
}

tasks = list(load_dataset("google-research-datasets/mbpp", "full", split="validation"))[: args.n]

# Two indexes: query=description only, and query=description+asserts.
indexes = {}
for name, with_tests in [("desc", False), ("tests", True)]:
    path = Path(f"indexes/mbpp_{name}")
    if not path.exists():
        Retriever.build(mbpp_corpus(with_tests=with_tests)).save(str(path))
    indexes[name] = Retriever.load(str(path))

model = ChatModel()
out_dir = Path("results/ablation")
out_dir.mkdir(parents=True, exist_ok=True)

summary = {}
for name, cfg in VARIANTS.items():
    cache = out_dir / f"{name}.json"
    if cache.exists():
        details = json.loads(cache.read_text(encoding="utf-8"))
    else:
        details = []
        for t in tasks:
            examples = []
            if cfg:
                idx, q_tests, k, floor, style = cfg
                q = mbpp_query(t["text"], t["test_list"] if q_tests else None)
                examples = [e for e in indexes[idx].search(q, k=k) if e["score"] >= floor]
            build = build_code_messages_reference if cfg and cfg[4] == "reference" else build_code_messages
            raw = model.chat(build(t["text"], t["test_list"], examples))
            res = execute_candidate(raw, t["test_list"])
            details.append({"task_id": t["task_id"], "passed": res.passed, "n_examples": len(examples),
                            "retrieved": [e["id"] for e in examples], "raw_output": raw})
            print(f"[{name}] {t['task_id']}: {'PASS' if res.passed else 'FAIL'} ({len(examples)} ex)", flush=True)
        cache.write_text(json.dumps(details, indent=1), encoding="utf-8")
    p = sum(d["passed"] for d in details)
    lo, hi = wilson_interval(p, len(details))
    summary[name] = (p, len(details), lo, hi)
    print(f"== {name}: {p}/{len(details)} = {p/len(details):.1%}  CI {lo:.1%}-{hi:.1%}", flush=True)

print("\n=== ABLATION (validation) ===")
base = {d["task_id"]: d["passed"] for d in json.loads((out_dir / "zero_shot.json").read_text(encoding="utf-8"))}
for name, (p, n, lo, hi) in summary.items():
    d = json.loads((out_dir / f"{name}.json").read_text(encoding="utf-8"))
    fixed = sum(1 for x in d if x["passed"] and not base[x["task_id"]])
    broken = sum(1 for x in d if not x["passed"] and base[x["task_id"]])
    print(f"{name:18s} {p}/{n} = {p/n:.1%}  CI {lo:.1%}-{hi:.1%}  vs zero-shot: +{fixed} fixed / -{broken} broken")
