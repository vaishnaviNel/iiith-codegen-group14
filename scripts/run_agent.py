"""Run the agent loop on MBPP test, reusing the zero-shot baseline as attempt 1.

Tasks the baseline already passed stay passed (attempt 1 passes -> loop stops). Only the baseline's
failures need new generations, which cuts the cost to ~half.

    python -m scripts.run_agent --limit 20        # pilot: first 20 failing tasks
    python -m scripts.run_agent                   # all failing tasks
"""

import argparse
import json
import time
from pathlib import Path

from datasets import load_dataset

from src.agent.loop import run_agent
from src.eval.code_eval import wilson_interval
from src.models.loader import ChatModel
from src.prompts import build_code_messages

ap = argparse.ArgumentParser()
ap.add_argument("--baseline", default="results/qwen_full_zero_shot.json")
ap.add_argument("--out", default="results/qwen_agent.json")
ap.add_argument("--max_attempts", type=int, default=3)
ap.add_argument("--limit", type=int, default=None, help="only retry this many failing tasks (pilot)")
args = ap.parse_args()

base = json.loads(Path(args.baseline).read_text(encoding="utf-8"))["zero_shot"]["details"]
base_by_id = {d["task_id"]: d for d in base}
tasks = {t["task_id"]: t for t in load_dataset("google-research-datasets/mbpp", "full", split="test")}
failing = [d["task_id"] for d in base if not d["passed"]]
todo = failing[: args.limit] if args.limit else failing

part = Path(args.out).with_suffix(".partial.json")
results = json.loads(part.read_text(encoding="utf-8")) if part.exists() else {}
model = ChatModel()

for tid in todo:
    if str(tid) in results:
        continue
    t = tasks[tid]
    t0 = time.time()
    r = run_agent(
        build_code_messages(t["text"], t["test_list"]), t["test_list"],
        chat=model.chat, max_attempts=args.max_attempts, first_output=base_by_id[tid]["raw_output"],
    )
    results[str(tid)] = {"passed": r.passed, "attempts": r.attempts, "trace": r.trace}
    part.write_text(json.dumps(results), encoding="utf-8")
    print(f"task {tid}: {'FIXED' if r.passed else 'still failing'} after {r.attempts} attempts "
          f"({time.time()-t0:.0f}s)", flush=True)

n_total = len(base)
already = sum(d["passed"] for d in base)
fixed = sum(1 for k in todo if results[str(k)]["passed"])
attempts_hist = {}
for k in todo:
    a = results[str(k)]["attempts"] if results[str(k)]["passed"] else "fail"
    attempts_hist[a] = attempts_hist.get(a, 0) + 1
print(f"\nRetried {len(todo)} failing tasks: fixed {fixed}  (fixed on attempt: {attempts_hist})")
if args.limit is None:
    p = already + fixed
    lo, hi = wilson_interval(p, n_total)
    print(f"AGENT pass rate with test feedback: {p}/{n_total} = {p/n_total:.1%}  (95% CI {lo:.1%}-{hi:.1%})")
    print(f"first-attempt pass@1 (baseline):    {already}/{n_total} = {already/n_total:.1%}")
    Path(args.out).write_text(json.dumps({"results": results, "pass_rate": p / n_total}), encoding="utf-8")
