"""The agent loop — plain Python, no framework.

RAG is one-shot:    prompt -> model -> answer.
An agent is a LOOP: prompt -> model -> act on the answer -> observe the result -> feed it back.

Here the "action" is running the code against the tests (a tool), and the "observation" is the
failure message. The whole agent is the `for` loop below; everything else is bookkeeping:

    messages = [system, task]
    for attempt in 1..max_attempts:
        answer = model(messages)             # think
        result = run_tests(answer)           # act (tool call)
        if result.passed: stop               # goal reached
        messages += [answer, feedback(result)]   # observe: model sees its own mistake next turn

The message list IS the agent's memory: each retry sees every earlier attempt and why it failed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from src.eval.code_eval import ExecResult, execute_candidate

Message = Dict[str, str]


def format_feedback(res: ExecResult) -> str:
    """Turn a failed ExecResult into the observation the model reads."""
    lines = [f"Your code failed ({res.error_type})."]
    if res.failed_test:
        lines.append(f"Failing test: {res.failed_test}")
    if res.error_message:
        lines.append(f"Details: {res.error_message[:600]}")  # cap: some asserts print huge inputs
    lines.append("Fix the bug. Reply with exactly one ```python code block containing the full corrected solution.")
    return "\n".join(lines)


@dataclass
class AgentResult:
    passed: bool
    attempts: int
    first_attempt_passed: bool
    final_output: str
    trace: List[Dict] = field(default_factory=list)  # one entry per attempt


def run_agent(
    messages: List[Message],
    tests: Sequence[str],
    chat: Callable[[List[Message]], str],
    max_attempts: int = 3,
    first_output: Optional[str] = None,
) -> AgentResult:
    """`chat` maps a message list to the model's reply. `first_output` lets a caller supply the
    first attempt when it is already known (greedy decoding is deterministic, so attempt 1 is
    identical to the zero-shot run and need not be regenerated)."""
    messages = list(messages)
    trace: List[Dict] = []
    output, res = "", None
    for attempt in range(1, max_attempts + 1):
        output = first_output if (attempt == 1 and first_output is not None) else chat(messages)
        res = execute_candidate(output, tests)
        trace.append({"attempt": attempt, "output": output, **res.to_dict()})
        if res.passed:
            break
        if attempt < max_attempts:
            messages += [
                {"role": "assistant", "content": output},
                {"role": "user", "content": format_feedback(res)},
            ]
    return AgentResult(
        passed=res.passed,
        attempts=len(trace),
        first_attempt_passed=trace[0]["passed"],
        final_output=output,
        trace=trace,
    )
