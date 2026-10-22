"""Evaluate Qwen on the first N MBPP test tasks, zero-shot vs RAG, fully local (CPU).

    python -m scripts.run_qwen_code --n 20
"""

import argparse
import json
import time
from pathlib import Path

from datasets import load_dataset

from src.eval.code_eval import execute_candidate, wilson_interval
from src.models.loader import ChatModel
from src.prompts import build_code_messages
from src.rag.retriever import Retriever

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=20)
ap.add_argument("--k", type=int, default=3, help="retrieved examples for RAG mode")
ap.add_argument("--modes", nargs="+", default=["zero_shot", "rag"])
ap.add_argument("--out", default="results/qwen_code.json")
args = ap.parse_args()

tasks = list(load_dataset("google-research-datasets/mbpp", "full", split="test"))[: args.n]
model = ChatModel()
retriever = Retriever.load("indexes/mbpp") if "rag" in args.modes else None

report = {}
for mode in args.modes:
    part = Path(args.out).with_suffix(f".{mode}.partial.json")  # resume after interruption
    details = json.loads(part.read_text(encoding="utf-8")) if part.exists() else []
    passed = sum(d["passed"] for d in details)
    done = {d["task_id"] for d in details}
    for t in tasks:
        if t["task_id"] in done:
            continue
        examples = retriever.search(t["text"], k=args.k) if mode == "rag" else []
        messages = build_code_messages(t["text"], t["test_list"], examples)
        t0 = time.time()
        raw = model.chat(messages)
        res = execute_candidate(raw, t["test_list"])
        passed += res.passed
        details.append({
            "task_id": t["task_id"], "passed": res.passed, "raw_output": raw,
            "error": getattr(res, "error_message", None), "secs": round(time.time() - t0, 1),
            "retrieved": [e["id"] for e in examples],
        })
        part.parent.mkdir(parents=True, exist_ok=True)
        part.write_text(json.dumps(details), encoding="utf-8")
        print(f"[{mode}] task {t['task_id']}: {'PASS' if res.passed else 'FAIL'}  ({details[-1]['secs']}s)", flush=True)
    lo, hi = wilson_interval(passed, len(tasks))
    report[mode] = {"pass_at_1": passed / len(tasks), "ci95": [lo, hi], "n": len(tasks), "details": details}
    print(f"== {mode}: pass@1 = {passed}/{len(tasks)} = {passed/len(tasks):.1%}  (95% CI {lo:.1%}-{hi:.1%})", flush=True)

Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
