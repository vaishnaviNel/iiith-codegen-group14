"""Docs agent: same loop as the code agent, but the "tool" is a static docstring check.

There are no unit tests for prose, so the check is a heuristic: the docstring must be non-empty and
must mention every parameter of the function. If not, the failure is fed back and the model retries.
"""

from __future__ import annotations

import ast
import re
import textwrap
from typing import Callable, Dict, List, Optional, Tuple

Message = Dict[str, str]


def clean_docstring(raw: str) -> str:
    """Model output -> plain docstring text (drop fences, triple quotes, a leading 'Docstring:')."""
    text = raw.strip()
    m = re.search(r"```(?:\w+)?\n(.*?)```", text, re.S)
    if m:
        text = m.group(1).strip()
    text = text.strip().strip('"').strip("'").strip()
    text = re.sub(r"^(docstring|summary)\s*:\s*", "", text, flags=re.I)
    return text.strip()


def function_params(code: str) -> List[str]:
    """Parameter names of the first function in `code` (self/cls excluded); [] if unparsable."""
    try:
        tree = ast.parse(textwrap.dedent(code))
    except SyntaxError:
        return []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            a = node.args
            names = [x.arg for x in a.posonlyargs + a.args + a.kwonlyargs]
            names += [a.vararg.arg] if a.vararg else []
            names += [a.kwarg.arg] if a.kwarg else []
            return [n for n in names if n not in ("self", "cls")]
    return []


def check_docstring(doc: str, code: str) -> Tuple[bool, str]:
    """(ok, feedback). Feedback is what the model reads on a retry."""
    if len(doc.split()) < 4:
        return False, "The docstring is empty or too short. Describe what the function does."
    missing = [p for p in function_params(code) if p not in doc]
    if missing:
        return False, (
            f"The docstring does not mention these parameters: {', '.join(missing)}. "
            "Describe each parameter by name."
        )
    return True, ""


def run_doc_agent(
    messages: List[Message],
    code: str,
    chat: Callable[[List[Message]], str],
    max_attempts: int = 3,
    first_output: Optional[str] = None,
) -> Dict:
    messages = list(messages)
    trace = []
    doc = ""
    for attempt in range(1, max_attempts + 1):
        raw = first_output if (attempt == 1 and first_output is not None) else chat(messages)
        doc = clean_docstring(raw)
        ok, feedback = check_docstring(doc, code)
        trace.append({"attempt": attempt, "doc": doc, "ok": ok, "feedback": feedback})
        if ok:
            break
        if attempt < max_attempts:
            messages += [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": feedback + " Reply with only the corrected docstring text."},
            ]
    return {"doc": doc, "attempts": len(trace), "first_ok": trace[0]["ok"], "ok": trace[-1]["ok"], "trace": trace}
