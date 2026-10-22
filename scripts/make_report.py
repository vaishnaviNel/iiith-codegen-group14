"""Collect every result file into REPORT.md. Missing files are reported as 'not run'."""

import json
from pathlib import Path

R = Path("results")


def load(name):
    p = R / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def pct(x):
    return f"{x:.1%}"


def ci(lo, hi):
    return f"{lo:.1%}–{hi:.1%}"


from src.eval.code_eval import wilson_interval  # noqa: E402

lines = ["# Final report — IIIT-H CodeGen Group 14", ""]

# ---------------- Task 1: text -> code ----------------
lines += ["## Task 1 — Text → Python (MBPP official test split)", ""]
base = load("qwen_full_zero_shot.json")
rows = []
base_pass = {}
if base:
    d = base["zero_shot"]["details"]
    base_pass = {x["task_id"]: x["passed"] for x in d}
    p = sum(base_pass.values())
    lo, hi = wilson_interval(p, len(d))
    rows.append(("Qwen2.5-Coder-1.5B-Instruct, zero-shot, pass@1", f"{p}/{len(d)} = {pct(p/len(d))}", ci(lo, hi), len(d)))
cg = load("codegen_code.json")
if cg:
    m = cg["metrics"]
    rows.append(("CodeGen-350M-multi, zero-shot, pass@1", pct(m["pass_at_1"]), ci(*m["pass_at_1_ci95"]), cg["n_samples"]))
ag = load("qwen_agent.json")
if ag and base:
    fixed = sum(1 for v in ag["results"].values() if v["passed"])
    p = sum(base_pass.values()) + fixed
    lo, hi = wilson_interval(p, len(base_pass))
    rows.append(("Qwen + agent loop (≤3 attempts, WITH test feedback)", f"{p}/{len(base_pass)} = {pct(p/len(base_pass))}", ci(lo, hi), len(base_pass)))
lines += ["| System | Score | 95% CI | n tasks |", "|---|---|---|---|"]
lines += [f"| {a} | {b} | {c} | {d} |" for a, b, c, d in rows] or ["| (no results) | | | |"]

sm = load("qwen_sampling.json")
if sm and ag and base:
    ids = sm["subset_ids"]
    any3 = sum(any(sm["passes"][str(i)]) for i in ids)
    agent_ok = sum(1 for i in ids if base_pass[i] or ag["results"].get(str(i), {}).get("passed"))
    greedy = sum(base_pass[i] for i in ids)
    n = len(ids)
    lines += ["", f"### Matched-budget comparison (same {n}-task random subset; up to 3 attempts each)", "",
              "| System | Solved | Rate | 95% CI |", "|---|---|---|---|"]
    for name, k in [("Greedy single attempt (no feedback)", greedy),
                    (f"Sampling, any of {sm['n']} samples (no feedback)", any3),
                    ("Agent loop, ≤3 attempts (test feedback)", agent_ok)]:
        lo, hi = wilson_interval(k, n)
        lines.append(f"| {name} | {k}/{n} | {pct(k/n)} | {ci(lo, hi)} |")
    lines += ["", f"Unbiased sampling estimate on this subset: pass@1 = {pct(sm['pass_at_1'])}, "
                  f"pass@{sm['n']} = {pct(sm['pass_at_%d' % sm['n']])} (T={sm['temperature']})."]

abl = sorted((R / "ablation").glob("*.json")) if (R / "ablation").exists() else []
if abl:
    lines += ["", "### RAG ablation (MBPP validation, first 40 tasks, Qwen)", "", "| Variant | pass@1 | 95% CI |", "|---|---|---|"]
    for f in abl:
        d = json.loads(f.read_text(encoding="utf-8"))
        p, n = sum(x["passed"] for x in d), len(d)
        lo, hi = wilson_interval(p, n)
        lines.append(f"| {f.stem} | {p}/{n} = {pct(p/n)} | {ci(lo, hi)} |")

if ag:
    hist = {}
    for v in ag["results"].values():
        k = f"fixed on attempt {v['attempts']}" if v["passed"] else "not fixed"
        hist[k] = hist.get(k, 0) + 1
    lines += ["", "Agent retries of the baseline's failures: " + ", ".join(f"{k}: {v}" for k, v in sorted(hist.items()))]

# ---------------- Task 2: code -> docs ----------------
lines += ["", "## Task 2 — Code → Documentation (CoDocBench test, first 100 samples, ROUGE-L F1)", ""]
dq = load("qwen_docs.json")
cgd = load("codegen_docs.json")
lines += ["| System | ROUGE-L | 95% CI | n |", "|---|---|---|---|"]
if cgd:
    m = cgd["metrics"]
    lines.append(f"| CodeGen-350M-multi, zero-shot | {m['rouge_l']:.4f} | {m['rouge_l_ci95'][0]:.3f}–{m['rouge_l_ci95'][1]:.3f} | {cgd['n_samples']} |")
names = {"zero_shot": "Qwen zero-shot", "rag": "Qwen + RAG (k=2 similar docstrings)", "rag_agent": "Qwen + RAG + check-and-retry agent"}
if dq:
    for mode, label in names.items():
        r = dq[mode]
        lines.append(f"| {label} | {r['rouge_l']:.4f} | {r['ci95'][0]:.3f}–{r['ci95'][1]:.3f} | {r['n']} |")
    lines += ["", "Docstring check (non-empty, mentions every parameter) pass rate: " +
              ", ".join(f"{names[m]} {dq[m]['check_pass_rate']:.0%}" for m in names)]
elif not cgd:
    lines.append("| (not run) | | | |")

lines += ["", "## Notes", "",
          "- Agent score uses the benchmark's visible test asserts as feedback (the standard MBPP prompt also shows them); compare it with the matched-budget sampling row, not with single-attempt pass@1.",
          "- RAG variants were tuned on the validation split; the test split was used only for final numbers.",
          "- CodeGen-350M baselines come from the separate zip pipeline (greedy, first n tasks/samples; see n); Qwen runs are bf16 on an RTX 4050.",
          "- Docs 'agent' checks structure (parameters mentioned), not correctness; ROUGE-L can move either way."]
Path("REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
