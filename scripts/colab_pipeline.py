"""CodeGen Group 14 pipeline — plain-Python port of colab/CodeGen_Group14_Colab.ipynb.

Same stages as the notebook (setup -> train -> evaluate code/docs -> summary), minus the
Colab-only parts (Drive mount, GPU assert, files.download, `!` magics). Every stage skips
itself if its output file already exists, so re-running resumes.

    python scripts/colab_pipeline.py --repo <project dir> --profile cpu_smoke
    python scripts/colab_pipeline.py --repo <project dir> --profile quick     # needs a GPU
    python scripts/colab_pipeline.py --repo <project dir> --profile full      # needs a GPU, hours

cpu_smoke: no training, tiny eval subsets (baseline + RAG + agent on the upstream model).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True, help="project dir containing src/train.py")
ap.add_argument("--out", default="codegen_outputs")
ap.add_argument("--profile", choices=["cpu_smoke", "quick", "full"], default="cpu_smoke")
ap.add_argument("--dataset", default="mbpp_full")
args = ap.parse_args()

REPO = Path(args.repo).resolve()
os.chdir(REPO)
PY = sys.executable
DATASET = args.dataset
UPSTREAM = "Salesforce/codegen-350M-multi"

# --- settings (the notebook's cell 0) -----------------------------------------------
TRAIN = args.profile != "cpu_smoke"
TRAIN_LORA = TRAIN
RUN_SAMPLING_BASELINE = TRAIN
EPOCHS = 3
AGENT_MAX_ATTEMPTS, AGENT_NUM_CANDIDATES = 3, 2
SAMPLING_N, PASS_K = 5, [1, 3, 5]

if args.profile == "cpu_smoke":
    MAX_TRAIN, MAX_CODE_EVAL, MAX_DOC_EVAL, MAX_VAL = None, 10, 5, None
elif args.profile == "quick":
    MAX_TRAIN, MAX_CODE_EVAL, MAX_DOC_EVAL, MAX_VAL, EPOCHS = 64, 40, 40, 20, 2
else:
    MAX_TRAIN, MAX_CODE_EVAL, MAX_DOC_EVAL, MAX_VAL = None, None, 500, None

RUN_TAG = f"{args.profile}_{DATASET}"
OUT = Path(args.out).resolve() / RUN_TAG
CKPT_OUT, RES_OUT = OUT / "checkpoints", OUT / "results"
RUNS = OUT / "runs"
for d in (CKPT_OUT, RES_OUT, RUNS):
    d.mkdir(parents=True, exist_ok=True)
print("Run tag:", RUN_TAG, "| out:", OUT, flush=True)


def run(cmd, done_marker=None):
    """Run a shell command, streaming output. Skip if done_marker exists."""
    if done_marker is not None and Path(done_marker).exists():
        print(f"SKIP (exists): {done_marker}")
        return
    print("$", cmd, flush=True)
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONPATH": str(REPO)}
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace", env=env)
    for line in proc.stdout:
        print(line, end="")
    if proc.wait() != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {cmd}")


def opt(flag, value):
    return f" {flag} {value}" if value is not None else ""


def q(p):
    return f'"{p}"'


# --- 1. setup -------------------------------------------------------------------------
import torch  # noqa: E402

print("CUDA available:", torch.cuda.is_available())
if TRAIN and not torch.cuda.is_available():
    sys.exit("Profiles 'quick'/'full' train models and need a GPU. Use --profile cpu_smoke here.")

run(f"{PY} data/scripts/download_and_preprocess.py", done_marker=REPO / "data/processed/codoc_test.jsonl")
run(f"{PY} -m pytest tests/ -q")

# --- 2. train -------------------------------------------------------------------------
FP16 = " --fp16" if torch.cuda.is_available() else ""
TRAIN_LIMIT = "" if MAX_TRAIN is None else f" --max_train_samples {MAX_TRAIN}"
FULL = " --full" if MAX_TRAIN is None else ""
COMMON_TRAIN = (f" --num_train_epochs {EPOCHS} --save_strategy epoch --save_total_limit {EPOCHS}"
                f"{FP16} --resume_from_checkpoint auto{FULL}{TRAIN_LIMIT}")


def train_and_select(task, kind):
    name = f"codegen-{'mbpp' if task == 'code' else 'doc'}-{kind}"
    best = CKPT_OUT / name
    if best.exists():
        print(f"SKIP (exists): {best}")
        return str(best)
    run_dir = RUNS / name
    script = "src/train.py" if task == "code" else "src/train_doc.py"
    extra = f" --dataset {DATASET}" if task == "code" else ""
    lora = " --lora" if kind == "lora" else ""
    bs = " --per_device_train_batch_size 2 --gradient_accumulation_steps 4"
    run(f"{PY} {script} --output_dir {q(run_dir)}{lora}{extra}{bs}{COMMON_TRAIN}")
    sel = f" --dataset {DATASET}" if task == "code" else ""
    run(f"{PY} -m src.eval.select_checkpoint --task {task} --run_dir {q(run_dir)} "
        f"--best_dir {q(best)}{sel}" + opt("--max_samples", MAX_VAL))
    return str(best)


CODE_FULL = DOC_FULL = CODE_LORA = DOC_LORA = None
if TRAIN:
    CODE_FULL = train_and_select("code", "full")
    CODE_LORA = train_and_select("code", "lora") if TRAIN_LORA else None
    DOC_FULL = train_and_select("docs", "full")
    DOC_LORA = train_and_select("docs", "lora") if TRAIN_LORA else None

# --- 3. evaluate: code (MBPP) ---------------------------------------------------------
CODE_N = opt("--max_samples", MAX_CODE_EVAL)


def code_eval(tag, model, extra=""):
    out = RES_OUT / f"code_{tag}.json"
    run(f"{PY} -m src.eval.run_eval --task code --dataset {DATASET} --split test "
        f"--model {model}{CODE_N}{extra} --output {q(out)}", done_marker=out)


code_eval("upstream", UPSTREAM)
code_eval("upstream_rag", UPSTREAM, " --rag")
if CODE_FULL:
    code_eval("finetune", q(CODE_FULL))
if CODE_LORA:
    code_eval("lora", q(CODE_LORA))
if RUN_SAMPLING_BASELINE and CODE_FULL:
    ks = " ".join(map(str, PASS_K))
    code_eval("finetune_sampling", q(CODE_FULL),
              f" --sampling_n {SAMPLING_N} --sampling_temperature 0.8 --pass_k {ks} --seed 0")
agent_model = q(CODE_FULL) if CODE_FULL else UPSTREAM
code_eval("agent", agent_model,
          f" --agent --max_attempts {AGENT_MAX_ATTEMPTS} --no_failure_context"
          f" --retry_temperature 0.8 --num_candidates {AGENT_NUM_CANDIDATES}")

# --- 4. evaluate: docs (CoDocBench) ---------------------------------------------------
DOC_N = opt("--max_samples", MAX_DOC_EVAL)


def doc_eval(tag, model):
    out = RES_OUT / f"docs_{tag}.json"
    run(f"{PY} -m src.eval.run_eval --task docs --split test --model {model}{DOC_N} --output {q(out)}",
        done_marker=out)


doc_eval("upstream", UPSTREAM)
if DOC_FULL:
    doc_eval("finetune", q(DOC_FULL))
if DOC_LORA:
    doc_eval("lora", q(DOC_LORA))

# --- 5. results -----------------------------------------------------------------------
files = " ".join(q(f) for f in sorted(RES_OUT.glob("*.json")))
run(f"{PY} experiments/summarize_results.py {files}")

rows = []
for f in sorted(RES_OUT.glob("*.json")):
    r = json.loads(f.read_text(encoding="utf-8"))
    m = r.get("metrics", {})
    if r["task"] == "code" and r["mode"] == "agent":
        rows.append((f.stem, "pass rate w/ test feedback", m["pass_rate_with_test_feedback"], r["n_samples"],
                     f"first-attempt pass@1={m['pass_at_1']:.1%}, samples/task={m['avg_samples_used']:.1f}"))
    elif r["task"] == "code" and r["mode"] == "sampling":
        rows += [(f.stem, f"pass@{k}", m[f"pass_at_{k}"], r["n_samples"], "") for k in PASS_K if f"pass_at_{k}" in m]
    elif r["task"] == "code":
        rows.append((f.stem, "pass@1", m["pass_at_1"], r["n_samples"], ""))
    else:
        rows.append((f.stem, "ROUGE-L", m["rouge_l"], r["n_samples"], f"AST fallback={m.get('ast_fallback_rate', 0):.0%}"))
print("\n=== SUMMARY ===")
for row in rows:
    print(f"{row[0]:28s} {row[1]:28s} {row[2]:.4f}  n={row[3]}  {row[4]}")
(RES_OUT / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
