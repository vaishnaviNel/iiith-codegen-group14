"""Prompt builders. Output is a chat message list for ChatModel.chat()."""

from __future__ import annotations

from typing import Dict, List, Sequence

SYSTEM_CODE = (
    "You are an expert Python programmer. Reply with exactly one ```python code block "
    "containing the complete solution (imports + function). No explanations, no tests, no example usage."
)


def _task_message(description: str, tests: Sequence[str]) -> str:
    # Showing the asserts is the standard MBPP protocol: it pins the function name and signature.
    if not tests:  # interactive use: the user may not have asserts
        return description
    return f"{description}\nYour code should pass these tests:\n" + "\n".join(tests)


SYSTEM_DOC = (
    "You are an expert Python developer. Write the docstring for the given function. "
    "Reply with only the docstring text: no code, no quotation marks, no markdown fences. "
    "Say what the function does, and describe its arguments and return value."
)


def build_doc_messages(code: str, examples: Sequence[Dict] = ()) -> List[Dict[str, str]]:
    """Code -> docstring. Retrieved (code, docstring) pairs go in the system prompt as style hints,
    since ROUGE-L against a project's own docstrings rewards matching that project's phrasing."""
    system = SYSTEM_DOC
    if examples:
        refs = "\n\n".join(
            f"# Code:\n{e['payload']['code'][:800]}\n# Docstring:\n{e['payload']['documentation'][:600]}"
            for e in examples
        )
        system += (
            "\n\nExamples of how similar functions are documented (match their style and length, "
            "but describe THIS function):\n\n" + refs
        )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Function:\n{code[:3000]}"},
    ]


REFERENCE_NOTE = (
    "\n\nBelow are solved problems retrieved by similarity. They may be unrelated or subtly "
    "different from the task. Use them only as style/idiom hints; always follow the task "
    "description and the tests exactly, and define every helper you use."
)


def build_code_messages_reference(
    description: str, tests: Sequence[str], examples: Sequence[Dict]
) -> List[Dict[str, str]]:
    """RAG as context, not as fake conversation: neighbours go in the system prompt with a
    warning, so the model is less inclined to copy a neighbour's logic wholesale."""
    refs = "\n\n".join(
        f"# Problem: {e['payload']['description']}\n{e['payload']['code'].strip()}" for e in examples
    )
    system = SYSTEM_CODE + (REFERENCE_NOTE + "\n\n" + refs if examples else "")
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": _task_message(description, tests)},
    ]


def build_code_messages(
    description: str, tests: Sequence[str], examples: Sequence[Dict] = ()
) -> List[Dict[str, str]]:
    """Zero-shot when `examples` is empty; with RAG, each retrieved MBPP hit becomes a prior
    user->assistant turn, so the model sees solved neighbours before the real task."""
    messages = [{"role": "system", "content": SYSTEM_CODE}]
    for ex in examples:
        p = ex["payload"]
        messages.append({"role": "user", "content": p["description"]})
        messages.append({"role": "assistant", "content": f"```python\n{p['code'].strip()}\n```"})
    messages.append({"role": "user", "content": _task_message(description, tests)})
    return messages
