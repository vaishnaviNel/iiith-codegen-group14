# IIIT-H CodeGen — Group 14

Two tasks, each with retrieval (RAG) and an agent loop:

1. **Text → Python code** (MBPP, metric: pass@1)
2. **Python code → documentation** (CoDocBench, metric: ROUGE-L)

Baseline: CodeGen-350M-multi. Main model: **Qwen2.5-Coder-1.5B-Instruct**. Results and tables: [REPORT.md](REPORT.md).

## Headline results

| | CodeGen-350M | Qwen zero-shot | Qwen + RAG | Qwen + agent |
|---|---|---|---|---|
| MBPP pass@1 (500 tasks) | 3.8% | 52.8% | no gain (ablation) | 56.4% (3 tries, test feedback) |
| CoDocBench ROUGE-L (100) | 0.160 | 0.196 | 0.227 | 0.224 |

At an equal budget of 3 tries, plain sampling (64.0% on a 150-task subset) beat the feedback agent (56.7%):
the agent's gain comes from extra attempts, not from reading the failure messages.

## Layout

```
src/rag/corpus.py       train-only corpora (MBPP train, CoDocBench train) + CoDocBench test loader
src/rag/retriever.py    dense retriever: bge-small embeddings, cosine top-k with numpy
src/models/loader.py    ChatModel.chat(messages) for Qwen; GPU bf16 if available, else CPU fp32
src/prompts.py          prompt builders (zero-shot, RAG as chat turns, RAG as reference block, docs)
src/agent/loop.py       code agent loop      src/agent/docs_agent.py   docs agent loop
src/eval/code_eval.py   sandboxed test runner, pass@k, Wilson / bootstrap CIs
scripts/                experiment drivers (see below)
tests/                  unit tests (no model needed)
```

## How to run

```bash
pip install -r requirements.txt            # plus a CUDA build of torch for the GPU
python -m scripts.build_index              # builds indexes/ (MBPP 374 entries, CoDocBench 2300)
python -m scripts.run_all --chain G        # every experiment, resumable; needs ../old_zip_pipeline for CodeGen
python -m scripts.make_report              # writes REPORT.md
python -m pytest tests -q
```

Demo UI (Gradio, runs locally, needs `indexes/` built first):

```bash
python -m ui.app          # open http://127.0.0.1:7860
```

Two tabs: **Text → Code** (zero-shot / RAG / agent with your own `assert` lines) and **Code → Docs**
(zero-shot / RAG / RAG + agent). Each shows every attempt, the test result and the retrieved examples.

Individual phases: `run_qwen_code` (baseline / RAG), `rag_ablation` (validation split),
`run_agent`, `run_sampling` (matched-budget baseline), `run_docs`. Each saves partial results
after every sample, so an interrupted run resumes.

## Retrieval (the "R" in RAG)

Every corpus entry is `{query, payload}`. We embed `query` once with `bge-small-en-v1.5` and keep a
normalised matrix. A search embeds the new task, takes dot products against the matrix (= cosine
similarity) and returns the top-k payloads. For a few thousand entries this is exact and instant, so no
vector database is needed. Only **train** splits are indexed, so test answers cannot leak.

## The agent loop (RAG → agent)

RAG is one pass: `prompt → model → answer`. An agent is a loop where the answer is **acted on** and the
result is fed back:

```
messages = [system, task]
for attempt in 1..max_attempts:
    answer = model(messages)               # think
    result = run_tests(answer)             # act: call a tool (sandboxed test runner)
    if result.passed: stop                 # goal reached
    messages += [answer, feedback(result)] # observe: the model sees its own mistake
```

The message list is the agent's memory: each retry sees every earlier attempt and why it failed
(`src/agent/loop.py`, ~60 lines, no framework). Design details:

* Attempt 1 equals the zero-shot answer (greedy decoding is deterministic), so only failures are retried.
* For docs there are no unit tests for prose, so the "tool" is a static check (`docs_agent.check_docstring`):
  non-empty and mentions every parameter; failures are fed back the same way.
* Agent scores use the benchmark's visible test asserts as feedback, so compare them with
  matched-budget sampling (pass@3), **not** with single-attempt pass@1.

## Notes

* RAG variants were tuned on the MBPP validation split; the test split is used only for final numbers.
* The CodeGen-350M baselines come from the earlier zip pipeline (`old_zip_pipeline`, kept separate).
* Small samples (40-task ablation, 150-task sampling subset, 100 docs samples) have wide confidence intervals.
