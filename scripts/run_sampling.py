"""Matched-budget baseline for the agent: n sampled solutions per task, NO test feedback.

The agent gets up to 3 attempts, so the fair comparison is pass@3 from 3 independent samples.
    python -m scripts.run_sampling --subset 150 --n 3
Resumable (partial saved per task).
"""

import argparse
import json
import random
from pathlib import Path

import torch
from datasets import load_dataset

from src.eval.code_eval import execute_candidate, pass_at_k, wilson_interval
from src.models.loader import ChatModel
from src.prompts import build_code_messages

ap = argparse.ArgumentParser()
ap.add_argument("--subset", type=int, default=150)
ap.add_argument("--n", type=int, default=3)
ap.add_argument("--temperature", type=float, default=0.8)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="results/qwen_sampling.json")
args = ap.parse_args()

all_tasks = list(load_dataset("google-research-datasets/mbpp", "full", split="test"))
ids = sorted(random.Random(args.seed).sample([t["task_id"] for t in all_tasks], args.subset))
by_id = {t["task_id"]: t for t in all_tasks}

part = Path(args.out).with_suffix(".partial.json")
done = json.loads(part.read_text(encoding="utf-8")) if part.exists() else {}
model = ChatModel()

for tid in ids:
    if str(tid) in done:
        continue
    t = by_id[tid]
    torch.manual_seed(args.seed * 100003 + tid)
    passes = []
    for _ in range(args.n):
        raw = model.chat(build_code_messages(t["text"], t["test_list"]), temperature=args.temperature)
        passes.append(execute_candidate(raw, t["test_list"]).passed)
    done[str(tid)] = passes
    part.write_text(json.dumps(done), encoding="utf-8")
    print(f"task {tid}: {sum(passes)}/{args.n} samples pass", flush=True)

out = {"subset_ids": ids, "n": args.n, "temperature": args.temperature, "passes": done}
for k in (1, args.n):
    out[f"pass_at_{k}"] = sum(pass_at_k(args.n, sum(done[str(i)]), k) for i in ids) / len(ids)
p3 = sum(any(done[str(i)]) for i in ids)
lo, hi = wilson_interval(p3, len(ids))
print(f"sampling pass@1 = {out['pass_at_1']:.1%}; any-of-{args.n} = {p3}/{len(ids)} = {p3/len(ids):.1%} (CI {lo:.1%}-{hi:.1%})")
Path(args.out).write_text(json.dumps(out), encoding="utf-8")
