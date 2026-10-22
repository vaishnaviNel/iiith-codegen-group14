"""Gradio demo for both tasks: text -> code and code -> docs, with RAG and the agent loop.

    python -m ui.app          # then open http://127.0.0.1:7860
"""

from __future__ import annotations

import threading

import gradio as gr

from src.agent.docs_agent import check_docstring, clean_docstring, run_doc_agent
from src.agent.loop import run_agent
from src.eval.code_eval import clean_generated_code, execute_candidate
from src.models.loader import ChatModel
from src.prompts import build_code_messages, build_code_messages_reference, build_doc_messages
from src.rag.retriever import Retriever

_lock = threading.Lock()  # one generation at a time: a single model on one GPU
_state: dict = {}


def model() -> ChatModel:
    if "model" not in _state:
        _state["model"] = ChatModel()
    return _state["model"]


def retriever(name: str) -> Retriever:
    if name not in _state:
        _state[name] = Retriever.load(f"indexes/{name}")
    return _state[name]


def _fmt_hits(hits, field_a: str, field_b: str) -> str:
    if not hits:
        return "_No retrieval (zero-shot)._"
    out = []
    for h in hits:
        p = h["payload"]
        out.append(f"**{h['id']}**  (similarity {h['score']:.3f})\n```python\n{p[field_a].strip()[:500]}\n```\n{p[field_b].strip()[:400]}")
    return "\n\n---\n\n".join(out)


def _attempt_md(n: int, output: str, res) -> str:
    head = "✅ passed" if res.passed else f"❌ failed — {res.error_type}"
    body = f"### Attempt {n}: {head}\n```python\n{clean_generated_code(output) or output}\n```"
    if not res.passed and (res.failed_test or res.error_message):
        body += f"\n**Failing test:** `{res.failed_test}`\n\n**Details:** {res.error_message[:500]}"
    return body


def run_code(task: str, tests_text: str, mode: str, k: int, max_attempts: int):
    if not task.strip():
        return "", "Enter a task description.", ""
    tests = [l.strip() for l in tests_text.splitlines() if l.strip()]
    if mode.startswith("Agent") and not tests:
        return "", "Agent mode needs at least one `assert` line: its tool is running your tests.", ""
    with _lock:
        hits = retriever("mbpp").search(task, k=k) if mode == "RAG" else []
        messages = (build_code_messages_reference if hits else build_code_messages)(task, tests, *( [hits] if hits else [] ))
        if mode.startswith("Agent"):
            r = run_agent(messages, tests, model().chat, max_attempts=max_attempts)
            trace = "\n\n".join(_attempt_md(t["attempt"], t["output"], _Res(t)) for t in r.trace)
            final, status = r.final_output, (f"✅ Passed after {r.attempts} attempt(s)" if r.passed
                                             else f"❌ Still failing after {r.attempts} attempts")
        else:
            raw = model().chat(messages)
            final = raw
            if tests:
                res = execute_candidate(raw, tests)
                trace, status = _attempt_md(1, raw, res), ("✅ Passed your tests" if res.passed else "❌ Failed your tests")
            else:
                trace, status = f"```python\n{clean_generated_code(raw) or raw}\n```", "Generated (no tests supplied, so not verified)"
    return clean_generated_code(final) or final, f"**{status}**\n\n{trace}", _fmt_hits(hits, "code", "description")


class _Res:
    """Adapter so trace dicts from the agent render like ExecResult."""

    def __init__(self, t):
        self.passed, self.error_type = t["passed"], t["error_type"]
        self.error_message, self.failed_test = t["error_message"], t["failed_test"]


def run_docs(code: str, mode: str, k: int, max_attempts: int):
    if not code.strip():
        return "", "Paste a Python function.", ""
    with _lock:
        hits = retriever("codoc").search(code, k=k) if mode != "Zero-shot" else []
        messages = build_doc_messages(code, hits)
        chat = lambda m: model().chat(m, max_new_tokens=220)  # noqa: E731
        if mode == "RAG + agent":
            r = run_doc_agent(messages, code, chat, max_attempts=max_attempts)
            doc = r["doc"]
            trace = "\n\n".join(
                f"### Attempt {t['attempt']}: {'✅ ok' if t['ok'] else '❌ ' + t['feedback']}\n{t['doc']}" for t in r["trace"])
        else:
            doc = clean_docstring(chat(messages))
            ok, fb = check_docstring(doc, code)
            trace = f"Check (non-empty, mentions every parameter): {'✅ ok' if ok else '❌ ' + fb}"
    return doc, trace, _fmt_hits(hits, "code", "documentation")


CODE_EXAMPLES = [
    ["Write a function to find the length of the shortest word in a list.",
     "assert len_shortest(['win', 'lose', 'great']) == 3\nassert len_shortest(['a', 'ab', 'abc']) == 1", "Agent (test feedback)", 3, 3],
    ["Write a function to check if a string is a palindrome.", "", "Zero-shot", 3, 3],
    ["Write a function to sort a matrix in ascending order by the sum of its rows.",
     "assert sort_matrix([[1, 2, 3], [2, 4, 5], [1, 1, 1]]) == [[1, 1, 1], [1, 2, 3], [2, 4, 5]]", "RAG", 2, 3],
]
DOC_EXAMPLE = ["def area(width, height, unit='cm'):\n    value = width * height\n    return f'{value} {unit}^2'", "RAG", 2, 3]

with gr.Blocks(title="CodeGen Group 14") as demo:
    gr.Markdown("# CodeGen Group 14\nQwen2.5-Coder-1.5B-Instruct with RAG and an agent loop. "
                "Everything runs locally; the first request is slower while the model warms up.")
    with gr.Tab("Text → Code"):
        with gr.Row():
            with gr.Column():
                task = gr.Textbox(label="Task description", lines=3)
                tests = gr.Textbox(label="Tests (optional, one `assert` per line; needed for Agent mode)", lines=4)
                mode = gr.Radio(["Zero-shot", "RAG", "Agent (test feedback)"], value="Zero-shot", label="Mode")
                with gr.Row():
                    k = gr.Slider(1, 5, value=2, step=1, label="Retrieved examples (RAG)")
                    att = gr.Slider(1, 5, value=3, step=1, label="Max attempts (Agent)")
                go = gr.Button("Generate", variant="primary")
            with gr.Column():
                out_code = gr.Code(label="Final code", language="python")
                out_trace = gr.Markdown(label="Run / attempts")
        with gr.Accordion("Retrieved examples", open=False):
            out_hits = gr.Markdown()
        go.click(run_code, [task, tests, mode, k, att], [out_code, out_trace, out_hits])
        gr.Examples(CODE_EXAMPLES, [task, tests, mode, k, att])
    with gr.Tab("Code → Docs"):
        with gr.Row():
            with gr.Column():
                code_in = gr.Code(label="Python function", language="python", lines=10)
                dmode = gr.Radio(["Zero-shot", "RAG", "RAG + agent"], value="RAG", label="Mode")
                with gr.Row():
                    dk = gr.Slider(1, 5, value=2, step=1, label="Retrieved examples")
                    datt = gr.Slider(1, 5, value=3, step=1, label="Max attempts (agent)")
                dgo = gr.Button("Generate docstring", variant="primary")
            with gr.Column():
                doc_out = gr.Textbox(label="Docstring", lines=8)
                doc_trace = gr.Markdown()
        with gr.Accordion("Retrieved similar functions", open=False):
            doc_hits = gr.Markdown()
        dgo.click(run_docs, [code_in, dmode, dk, datt], [doc_out, doc_trace, doc_hits])
        gr.Examples([DOC_EXAMPLE], [code_in, dmode, dk, datt])

if __name__ == "__main__":
    model()  # load once up front so a bad install fails here, not on the first click
    demo.launch(server_name="127.0.0.1", server_port=7860)
