"""Run the remaining phases. Two chains can run in parallel (4 CPU threads each):

    OMP_NUM_THREADS=4 python -m scripts.run_all --chain A     # code agent -> sampling comparison
    OMP_NUM_THREADS=4 python -m scripts.run_all --chain B     # docs -> CodeGen-350M baselines
    python -m scripts.make_report                              # when both are done

Every step is resumable and skipped when its final output exists.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--chain", choices=["A", "B", "G"], required=True)
ap.add_argument("--old_repo", default=r"C:\capstone_project\old_zip_pipeline", help="the zip project (CodeGen baseline)")
ap.add_argument("--codegen_code_n", type=int, default=200)
ap.add_argument("--codegen_docs_n", type=int, default=100)
args = ap.parse_args()

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
ENV = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONPATH": str(ROOT)}


def step(name, cmd, out, cwd=ROOT, env=None):
    if Path(out).exists():
        print(f"SKIP {name} (exists: {out})", flush=True)
        return
    print(f"\n##### {name}\n$ {' '.join(map(str, cmd))}", flush=True)
    rc = subprocess.call(list(map(str, cmd)), cwd=cwd, env=env or ENV)
    if rc != 0:
        print(f"!!! {name} failed with exit code {rc}; continuing with the next step", flush=True)


py = sys.executable
if args.chain == "G":
    # Single GPU chain (6 GB VRAM: run one model process at a time).
    step("Qwen zero-shot baseline, 500 MBPP tasks",
         [py, "-m", "scripts.run_qwen_code", "--n", 500, "--modes", "zero_shot", "--out", R / "qwen_full_zero_shot.json"],
         R / "qwen_full_zero_shot.json")
    step("RAG ablation (validation)", [py, "-m", "scripts.rag_ablation", "--n", 40],
         R / "ablation" / "ref_k2_tests_t75.json")
    step("code agent (retry all baseline failures)", [py, "-m", "scripts.run_agent"], R / "qwen_agent.json")
    step("matched-budget sampling baseline", [py, "-m", "scripts.run_sampling", "--subset", 150, "--n", 3],
         R / "qwen_sampling.json")
    step("docs: Qwen zero-shot / RAG / RAG+agent", [py, "-m", "scripts.run_docs", "--n", 100], R / "qwen_docs.json")
    old = Path(args.old_repo)
    step("CodeGen-350M code baseline",
         [py, "-m", "src.eval.run_eval", "--task", "code", "--dataset", "mbpp_full", "--split", "test",
          "--model", "Salesforce/codegen-350M-multi", "--max_samples", 500,
          "--output", R / "codegen_code.json"], R / "codegen_code.json", cwd=old)
    step("CodeGen-350M docs baseline",
         [py, "-m", "src.eval.run_eval", "--task", "docs", "--split", "test",
          "--model", "Salesforce/codegen-350M-multi", "--max_samples", 100,
          "--output", R / "codegen_docs.json"], R / "codegen_docs.json", cwd=old)
    print("\nCHAIN G DONE", flush=True)
    sys.exit(0)
if args.chain == "A":
    # Attempt-1 results of the agent are the zero-shot baseline; reuse the pilot's retries too.
    pilot, full = R / "qwen_agent_pilot.partial.json", R / "qwen_agent.partial.json"
    if pilot.exists() and not full.exists():
        shutil.copy(pilot, full)
    step("code agent (retry all baseline failures)", [py, "-m", "scripts.run_agent"], R / "qwen_agent.json")
    step("matched-budget sampling baseline", [py, "-m", "scripts.run_sampling", "--subset", 150, "--n", 3],
         R / "qwen_sampling.json")
else:
    step("docs: Qwen zero-shot / RAG / RAG+agent", [py, "-m", "scripts.run_docs", "--n", 100], R / "qwen_docs.json")
    old = Path(args.old_repo)
    step("CodeGen-350M code baseline",
         [py, "-m", "src.eval.run_eval", "--task", "code", "--dataset", "mbpp_full", "--split", "test",
          "--model", "Salesforce/codegen-350M-multi", "--max_samples", args.codegen_code_n,
          "--output", R / "codegen_code.json"], R / "codegen_code.json", cwd=old)
    step("CodeGen-350M docs baseline",
         [py, "-m", "src.eval.run_eval", "--task", "docs", "--split", "test",
          "--model", "Salesforce/codegen-350M-multi", "--max_samples", args.codegen_docs_n,
          "--output", R / "codegen_docs.json"], R / "codegen_docs.json", cwd=old)
print(f"\nCHAIN {args.chain} DONE", flush=True)
