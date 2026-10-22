# Project context (read this first — handoff from the previous machine)

IIIT-H CodeGen capstone, **Group 14**. Two tasks, each with RAG + an agent loop:
1. Text → Python (MBPP, pass@1)   2. Python → docs (CoDocBench, ROUGE-L)

Main model: **Qwen2.5-Coder-1.5B-Instruct**. Baseline: CodeGen-350M-multi (measured with an older
separate pipeline that is NOT in this repo). Results: `REPORT.md`. Layout and commands: `README.md`.

## How the user wants to work
- Terse communication; make reasonable inferences instead of asking many questions.
- User is an ML compiler engineer, comfortable with RAG, **new to agents** — explain agent concepts starting
  from the RAG baseline. When guiding a build, give complete code per step and explain each step (no fill-in-the-blank).
- The agent is **plain Python, not LangGraph** (user wants to understand the loop; see `src/agent/loop.py`).
- **Git: never put the user's personal name/email in commits.** Use the neutral identity
  `Group 14 <group14@users.noreply.github.com>` (set per-repo with `git config user.name/user.email`; check before every commit).
  Repo is shared with a collaborator. Do not push anything without being asked. Check results files for the OS username before committing.
- Ask before installing software or doing outward-facing/irreversible things.

## State when handed off
Done: retriever + indexes, Qwen loader, prompts, code agent, docs agent, sandboxed evaluator, RAG ablation,
matched-budget sampling baseline, docs experiments, CodeGen baselines, Gradio UI, README, REPORT.md, 11 unit tests.
Not done / open: (a) UI + README + CLAUDE.md changes were not yet committed (one local commit exists from before
the UI), (b) nothing has been pushed, (c) no extra work queued — ask the user what's next.

## Final results (Qwen runs: bf16 on an RTX 4050; MBPP official test = 500 tasks)
| | CodeGen-350M | Qwen zero-shot | Qwen + RAG | Qwen + agent |
|---|---|---|---|---|
| MBPP pass@1 | 3.8% (CI 2.4–5.9) | 52.8% (48.4–57.1) | no reliable gain (40-task validation ablation: 37.5–47.5% vs 42.5% zero-shot) | 56.4% (52.0–60.7), ≤3 attempts WITH test feedback; fixed 18 of 236 failures |
| CoDocBench ROUGE-L (100 samples) | 0.160 | 0.196 | 0.227 | 0.224 (parameter-mention check 74% → 88%, ROUGE-L unchanged) |

Key finding: at an equal budget of 3 tries on a 150-task subset, plain sampling (64.0%) beat the feedback agent (56.7%)
and greedy single attempt (53.3%) — the agent's gain is extra attempts, not use of failure messages.
RAG helps docs (+0.03 ROUGE-L) but not code. Caveats: small samples → wide CIs; CodeGen docs baseline used the old pipeline's prompt.

## Gotchas for the new machine
- The previous machine was Windows + RTX 4050 (6 GB). `ChatModel` uses CUDA bf16 if available, else CPU fp32
  (~10–20× slower: ~20 s/task). On a Mac without CUDA everything still runs, just slowly; Colab is the GPU fallback
  (`scripts/colab_pipeline.py` is an older port of the original notebook for the *old* pipeline).
- `pip install -r requirements.txt` installs the CPU/default torch; for NVIDIA install the CUDA wheel
  (`pip install --index-url https://download.pytorch.org/whl/cu128 torch`).
- Rebuild indexes first: `python -m scripts.build_index` (also builds nothing for `mbpp_desc`/`mbpp_tests`; those two are
  created on demand by `scripts/rag_ablation.py`). The UI needs `indexes/mbpp` and `indexes/codoc`.
- `python -m scripts.run_all --chain G` runs every experiment (all steps resumable, skip if output exists) but its last two
  steps need the old CodeGen pipeline at `--old_repo` (not included); results for those are already in `results/`.
  Re-running Qwen steps on a different device/dtype will change numbers slightly (greedy outputs can flip).
- Run UI: `python -m ui.app` → http://127.0.0.1:7860. Run tests: `python -m pytest tests -q`.
- Scripts are run as modules from the repo root (`python -m scripts.<name>`).
- `results/*.partial.json` are resume files (git-ignored). Logs are git-ignored.

## Possible next steps (none started)
Full 500-task RAG run of the best variant (1 example, similarity ≥ 0.80); larger docs eval (more than 100 samples);
better docs agent check (not just "mentions parameters"); fix retry feedback so the model stops over-correcting
(see agent traces in `results/qwen_agent.json`); try a larger model or sampling-based agent (best-of-n + tests).
