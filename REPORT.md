# Final report — IIIT-H CodeGen Group 14

## Task 1 — Text → Python (MBPP official test split)

| System | Score | 95% CI | n tasks |
|---|---|---|---|
| Qwen2.5-Coder-1.5B-Instruct, zero-shot, pass@1 | 264/500 = 52.8% | 48.4%–57.1% | 500 |
| CodeGen-350M-multi, zero-shot, pass@1 | 3.8% | 2.4%–5.9% | 500 |
| Qwen + agent loop (≤3 attempts, WITH test feedback) | 282/500 = 56.4% | 52.0%–60.7% | 500 |

### Matched-budget comparison (same 150-task random subset; up to 3 attempts each)

| System | Solved | Rate | 95% CI |
|---|---|---|---|
| Greedy single attempt (no feedback) | 80/150 | 53.3% | 45.4%–61.1% |
| Sampling, any of 3 samples (no feedback) | 96/150 | 64.0% | 56.1%–71.2% |
| Agent loop, ≤3 attempts (test feedback) | 85/150 | 56.7% | 48.7%–64.3% |

Unbiased sampling estimate on this subset: pass@1 = 49.8%, pass@3 = 64.0% (T=0.8).

### RAG ablation (MBPP validation, first 40 tasks, Qwen)

| Variant | pass@1 | 95% CI |
|---|---|---|
| ref_k2_tests | 18/40 = 45.0% | 30.7%–60.2% |
| ref_k2_tests_t75 | 18/40 = 45.0% | 30.7%–60.2% |
| ref_k3 | 17/40 = 42.5% | 28.5%–57.8% |
| turns_k1_t80 | 19/40 = 47.5% | 32.9%–62.5% |
| turns_k3 | 15/40 = 37.5% | 24.2%–53.0% |
| zero_shot | 17/40 = 42.5% | 28.5%–57.8% |

Agent retries of the baseline's failures: fixed on attempt 2: 15, fixed on attempt 3: 3, not fixed: 218

## Task 2 — Code → Documentation (CoDocBench test, first 100 samples, ROUGE-L F1)

| System | ROUGE-L | 95% CI | n |
|---|---|---|---|
| CodeGen-350M-multi, zero-shot | 0.1601 | 0.142–0.179 | 100 |
| Qwen zero-shot | 0.1960 | 0.174–0.219 | 100 |
| Qwen + RAG (k=2 similar docstrings) | 0.2269 | 0.205–0.249 | 100 |
| Qwen + RAG + check-and-retry agent | 0.2240 | 0.202–0.246 | 100 |

Docstring check (non-empty, mentions every parameter) pass rate: Qwen zero-shot 85%, Qwen + RAG (k=2 similar docstrings) 74%, Qwen + RAG + check-and-retry agent 88%

## Notes

- Agent score uses the benchmark's visible test asserts as feedback (the standard MBPP prompt also shows them); compare it with the matched-budget sampling row, not with single-attempt pass@1.
- RAG variants were tuned on the validation split; the test split was used only for final numbers.
- CodeGen-350M baselines come from the separate zip pipeline (greedy, first n tasks/samples; see n); Qwen runs are bf16 on an RTX 4050.
- Docs 'agent' checks structure (parameters mentioned), not correctness; ROUGE-L can move either way.
