"""Code generation evaluation: execution accuracy and pass@1."""

from __future__ import annotations

import ast
import inspect
import json
import math
import os
import random
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


DEFAULT_TIMEOUT_SECONDS = 5
DEFAULT_MEMORY_LIMIT_MB = 1024


def _limit_resources(memory_mb: int):
    """Return a preexec_fn that caps memory/CPU for the child (POSIX only)."""
    def _apply():  # pragma: no cover - runs in the child process
        try:
            import resource

            limit = memory_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        except Exception:
            pass
        os.setsid()

    return _apply


def run_in_sandbox(
    script: str,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    memory_mb: int = DEFAULT_MEMORY_LIMIT_MB,
) -> bool:
    """
    Execute untrusted model-generated code in a separate Python process.

    - Runs `python -I` (isolated mode: ignores PYTHON* env vars and user site dir)
      inside a fresh temporary working directory, so stray file writes land there.
    - Hard wall-clock timeout via subprocess (works from any thread -- unlike the
      old signal.alarm approach, which silently disabled the timeout under FastAPI).
    - Memory cap via RLIMIT_AS on Linux/macOS.

    This is process isolation, not a security boundary against a malicious actor;
    for a public deployment run it inside a container/gVisor/firejail as well.
    Returns True iff the script exits with status 0.
    """
    with tempfile.TemporaryDirectory(prefix="codegen_exec_") as tmp:
        script_path = Path(tmp) / "candidate.py"
        script_path.write_text(script, encoding="utf-8")
        kwargs = {}
        if os.name == "posix":
            kwargs["preexec_fn"] = _limit_resources(memory_mb)
        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(script_path)],
                cwd=tmp,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=timeout_seconds,
                env={"PATH": os.environ.get("PATH", ""), "PYTHONHASHSEED": "0"},
                **kwargs,
            )
        except subprocess.TimeoutExpired:
            return False
        except OSError:
            return False
        return proc.returncode == 0


# ---------------------------------------------------------------------------
# Structured execution results (Phase 1): WHY did a candidate fail?
# ---------------------------------------------------------------------------

# Error categories reported for every executed candidate.
ERROR_TYPES = (
    "Passed",          # all asserts passed
    "AssertionError",  # code ran, but produced a wrong answer
    "NameError",       # wrong/undefined function or variable name (often a prompt/harness issue)
    "ImportError",     # missing / unavailable module
    "SyntaxError",     # unparsable code (often truncated output)
    "Timeout",         # infinite loop or far too slow
    "MemoryError",     # exceeded the sandbox memory cap
    "RuntimeError",    # any other exception (TypeError, IndexError, RecursionError, ...)
    "EmptyOutput",     # model produced no code at all
    "Crash",           # child process died without reporting (segfault, killed, ...)
)


@dataclass
class ExecResult:
    passed: bool
    error_type: str = "Passed"
    error_message: str = ""
    failed_test: Optional[str] = None  # the assert line that failed, if any

    def to_dict(self) -> Dict:
        return {
            "passed": self.passed,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "failed_test": self.failed_test,
        }


# The harness runs INSIDE the sandboxed child process. It executes the candidate
# code, then each test line one by one, and writes a small JSON report to a file
# (not stdout, so prints from the generated code can't corrupt it).
_HARNESS = r'''
import ast, json, sys, traceback

report_path, code_path, tests_path = sys.argv[1], sys.argv[2], sys.argv[3]

def report(**kw):
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(kw, f)

def classify(exc):
    if isinstance(exc, AssertionError):
        return "AssertionError"
    if isinstance(exc, (ImportError, ModuleNotFoundError)):
        return "ImportError"
    if isinstance(exc, NameError):
        return "NameError"
    if isinstance(exc, MemoryError):
        return "MemoryError"
    if isinstance(exc, SyntaxError):
        return "SyntaxError"
    return "RuntimeError"

def short_tb(exc):
    # "NameError: name 'f' is not defined" first, then where it happened in the
    # candidate/test code (harness frames are hidden).
    head = "".join(traceback.format_exception_only(type(exc), exc)).strip()
    frames = [f for f in traceback.extract_tb(exc.__traceback__) if not f.filename.endswith("harness.py")]
    where = "; ".join(f"{f.filename}:{f.lineno} in {f.name}" for f in frames[-2:])
    return (head + (f"  [at {where}]" if where else ""))[:800]

def describe_assert(test_src, env):
    # For `assert f(x) == y`, show what f(x) actually returned.
    try:
        node = ast.parse(test_src.strip()).body[0]
        if isinstance(node, ast.Assert) and isinstance(node.test, ast.Compare) \
                and len(node.test.ops) == 1 and isinstance(node.test.ops[0], ast.Eq):
            left = ast.unparse(node.test.left)
            right = ast.unparse(node.test.comparators[0])
            got = eval(compile(ast.Expression(node.test.left), "<test>", "eval"), env)
            return f"{left} returned {got!r}, expected {right}"
    except BaseException:
        pass
    return "assertion failed"

code = open(code_path, encoding="utf-8").read()
tests = json.load(open(tests_path, encoding="utf-8"))

try:
    compiled = compile(code, "candidate.py", "exec")
except SyntaxError as exc:
    report(passed=False, error_type="SyntaxError",
           error_message=f"{exc.msg} (line {exc.lineno}): {(exc.text or '').strip()}"[:600],
           failed_test=None)
    sys.exit(0)

env = {"__name__": "__candidate__"}
try:
    exec(compiled, env)
except BaseException as exc:
    report(passed=False, error_type=classify(exc), error_message=short_tb(exc), failed_test=None)
    sys.exit(0)

for test in tests:
    if not test.strip():
        continue
    try:
        exec(compile(test, "<test>", "exec"), env)
    except AssertionError:
        report(passed=False, error_type="AssertionError",
               error_message=describe_assert(test, env), failed_test=test.strip())
        sys.exit(0)
    except BaseException as exc:
        report(passed=False, error_type=classify(exc), error_message=short_tb(exc),
               failed_test=test.strip())
        sys.exit(0)

report(passed=True, error_type="Passed", error_message="", failed_test=None)
'''


def execute_with_tests(
    code: str,
    test_list: Sequence[str],
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    memory_mb: int = DEFAULT_MEMORY_LIMIT_MB,
) -> ExecResult:
    """
    Run `code` and then each test line in a sandboxed subprocess; return WHY it failed.

    Same isolation as run_in_sandbox (python -I, temp cwd, hard timeout, memory cap).
    """
    if not (code or "").strip():
        return ExecResult(False, "EmptyOutput", "model produced no code")
    with tempfile.TemporaryDirectory(prefix="codegen_exec_") as tmp:
        tmp_path = Path(tmp)
        harness = tmp_path / "harness.py"
        code_file = tmp_path / "candidate.py"
        tests_file = tmp_path / "tests.json"
        report_file = tmp_path / "report.json"
        harness.write_text(_HARNESS, encoding="utf-8")
        code_file.write_text(code, encoding="utf-8")
        tests_file.write_text(json.dumps(list(test_list)), encoding="utf-8")
        kwargs = {}
        if os.name == "posix":
            kwargs["preexec_fn"] = _limit_resources(memory_mb)
        try:
            subprocess.run(
                [sys.executable, "-I", str(harness), str(report_file), str(code_file), str(tests_file)],
                cwd=tmp,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=timeout_seconds,
                env={"PATH": os.environ.get("PATH", ""), "PYTHONHASHSEED": "0"},
                **kwargs,
            )
        except subprocess.TimeoutExpired:
            return ExecResult(False, "Timeout", f"exceeded {timeout_seconds}s")
        except OSError as exc:
            return ExecResult(False, "Crash", f"could not start sandbox: {exc}")
        if not report_file.exists():
            return ExecResult(False, "Crash", "sandbox process exited without a report")
        try:
            data = json.loads(report_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return ExecResult(False, "Crash", f"unreadable sandbox report: {exc}")
        return ExecResult(
            passed=bool(data.get("passed")),
            error_type=data.get("error_type", "Crash"),
            error_message=data.get("error_message", ""),
            failed_test=data.get("failed_test"),
        )


# ---------------------------------------------------------------------------
# Cleaning model output into runnable code
# ---------------------------------------------------------------------------

_FENCE = re.compile(r"```[ \t]*([A-Za-z0-9_+-]*)[ \t]*\n(.*?)(?:```|\Z)", re.DOTALL)


def _extract_fenced_code(text: str) -> Optional[str]:
    """If the output contains markdown code blocks (chat models), return the best one."""
    blocks = [(lang.lower(), body) for lang, body in _FENCE.findall(text)]
    if not blocks:
        return None
    for lang, body in blocks:  # prefer a python block that defines something
        if lang in ("python", "py", "python3", "") and re.search(r"^\s*(def|class) ", body, re.M):
            return body
    for _, body in blocks:
        if re.search(r"^\s*(def|class) ", body, re.M):
            return body
    return blocks[0][1]


def _parse_longest_prefix(lines: List[str]) -> Optional[Tuple[ast.Module, List[str]]]:
    """Parse the code; if the end is truncated, drop trailing lines until it parses."""
    for end in range(len(lines), 0, -1):
        chunk = lines[:end]
        try:
            return ast.parse("\n".join(chunk)), chunk
        except SyntaxError:
            continue
    return None


def _is_constant_assign(node: ast.AST) -> bool:
    """Top-level `NAME = <literal>` (e.g. MOD = 10**9 + 7) is kept; anything that calls code is not."""
    if not isinstance(node, (ast.Assign, ast.AnnAssign)):
        return False
    value = node.value
    if value is None:
        return False
    return not any(isinstance(n, (ast.Call, ast.Await, ast.Yield)) for n in ast.walk(value))


def _legacy_first_function(lines: List[str]) -> str:
    """Old behaviour, used only when the code can't be parsed at all."""
    first_def = next(
        (i for i, line in enumerate(lines) if line.startswith("def ") or line.startswith("class ")),
        None,
    )
    if first_def is None:
        return "\n".join(lines).strip()
    imports = [l for l in lines[:first_def] if l.startswith("import ") or l.startswith("from ")]
    body: List[str] = []
    for offset, line in enumerate(lines[first_def:]):
        if offset > 0 and line and not line[0].isspace():
            break
        body.append(line)
    return "\n".join(imports + body).strip()


def clean_generated_code(raw_code: str) -> str:
    """
    Turn raw model output into the code we execute.

    KEEP: imports, every top-level function/class (helper functions included), and
          constant assignments.
    DROP: markdown fences / prose (chat models), top-level asserts, prints, loops,
          `if __name__ == "__main__":` blocks, and any re-definition of a function
          name that was already defined (models sometimes write the same function twice;
          the later copy would silently override the first).

    The old version kept only the FIRST function, so any correct solution that used a
    helper function (e.g. `is_prime` called by the main function) failed with NameError.
    """
    text = raw_code or ""
    fenced = _extract_fenced_code(text)
    if fenced is not None:
        text = fenced
    lines = [line for line in text.replace("\r\n", "\n").split("\n") if not line.strip().startswith("```")]
    if not any(l.strip() for l in lines):
        return ""

    parsed = _parse_longest_prefix(lines)
    if parsed is None:
        return _legacy_first_function(lines)
    tree, src_lines = parsed

    kept: List[str] = []
    seen_defs = set()
    for node in tree.body:
        start = node.lineno
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name in seen_defs:
                continue
            seen_defs.add(node.name)
            if node.decorator_list:
                start = min(d.lineno for d in node.decorator_list)
        elif isinstance(node, (ast.Import, ast.ImportFrom)) or _is_constant_assign(node):
            pass
        else:
            continue  # asserts, prints, loops, __main__ guards, calls ...
        kept.append("\n".join(src_lines[start - 1 : node.end_lineno]))

    if not seen_defs:
        # Nothing callable survived: fall back to old behaviour so we don't hide output.
        return _legacy_first_function(lines)
    return "\n\n".join(kept).strip()


def evaluate_functional_correctness(
    generated_code: str,
    test_list: List[str],
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> bool:
    """Return True if generated code passes all assert tests (run in a sandboxed subprocess)."""
    return execute_candidate(generated_code, test_list, timeout_seconds).passed


def execute_candidate(
    generated_code: str,
    test_list: List[str],
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> ExecResult:
    """Clean the generated code, then run it against the tests with a structured result."""
    return execute_with_tests(clean_generated_code(generated_code), test_list, timeout_seconds)


def pass_at_1(results: List[bool]) -> float:
    if not results:
        return 0.0
    return sum(1 for ok in results if ok) / len(results)


def pass_at_k(n: int, c: int, k: int) -> float:
    """
    Unbiased pass@k estimator (Chen et al., 2021, "Evaluating LLMs Trained on Code").

    n = samples drawn for a task, c = how many of them passed, k <= n.
    """
    if k > n:
        raise ValueError(f"k={k} must be <= n={n}")
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


def wilson_interval(successes: int, total: int, z: float = 1.96) -> Tuple[float, float]:
    """95% Wilson score interval for a binomial proportion (e.g. pass@1 over n tasks)."""
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def bootstrap_ci(
    values: Sequence[float],
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[float, float]:
    """Percentile bootstrap CI for the mean of per-example scores (ROUGE-L, pass@k, ...)."""
    vals = list(values)
    if not vals:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(vals)
    means = sorted(sum(rng.choice(vals) for _ in range(n)) / n for _ in range(n_resamples))
    lo = means[int((alpha / 2) * n_resamples)]
    hi = means[min(n_resamples - 1, int((1 - alpha / 2) * n_resamples))]
    return (lo, hi)


def compute_codebleu(pairs: List[tuple], lang: str = "python") -> Optional[float]:
    """Average CodeBLEU over (reference, prediction) pairs; None if codebleu isn't installed."""
    try:
        from codebleu import calc_codebleu
    except ImportError:
        return None
    scores = []
    for ref, pred in pairs:
        if not (ref or "").strip() or not (pred or "").strip():
            continue
        try:
            result = calc_codebleu([ref], [pred], lang=lang)
            scores.append(result["codebleu"])
        except Exception:
            # Malformed generations can trip the AST/dataflow matchers; skip rather than fail the run.
            continue
    return sum(scores) / len(scores) if scores else None


def _accepts_tests(generate_fn) -> bool:
    """Whether generate_fn takes the test list as a second argument."""
    try:
        params = inspect.signature(generate_fn).parameters
    except (TypeError, ValueError):
        return False
    if any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in params.values()):
        return True
    positional = [
        p
        for p in params.values()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    return len(positional) >= 2


def evaluate_code_batch(
    examples: List[Dict],
    generate_fn,
    max_samples: Optional[int] = None,
) -> Dict:
    """
    Run pass@1 over a list of MBPP-style examples.

    generate_fn may return raw model output; it is cleaned here, and the raw text is
    kept in the details so failures can be inspected (experiments/analyze_failures.py).
    """
    import time

    subset = examples[:max_samples] if max_samples else examples
    pass_tests = _accepts_tests(generate_fn)
    per_task = []
    for ex in subset:
        prompt = ex.get("prompt", "")
        tests = ex.get("test_list") or []
        if not tests and ex.get("test_template"):
            tests = [line for line in ex["test_template"].split("\n") if line.strip()]
        t0 = time.time()
        generated = generate_fn(prompt, tests) if pass_tests else generate_fn(prompt)
        gen_seconds = time.time() - t0
        cleaned = clean_generated_code(generated)
        if tests:
            res = execute_with_tests(cleaned, tests)
        else:
            res = ExecResult(False, "RuntimeError", "task has no tests")
        per_task.append(
            {
                "id": ex.get("id"),
                "prompt": prompt,
                "tests": tests,
                "raw_output": generated,
                "generated_code": cleaned,
                "reference_code": ex.get("code") or ex.get("canonical_solution") or "",
                "gen_seconds": round(gen_seconds, 3),
                **res.to_dict(),
            }
        )
    n_passed = sum(1 for r in per_task if r["passed"])
    score = pass_at_1([r["passed"] for r in per_task])
    codebleu = compute_codebleu([(r["reference_code"], r["generated_code"]) for r in per_task])
    return {
        "pass_at_1": score,
        "pass_at_1_ci95": wilson_interval(n_passed, len(per_task)),
        "codebleu": codebleu,
        "n_samples": len(per_task),
        "n_passed": n_passed,
        "error_breakdown": error_breakdown(per_task),
        "details": per_task,
    }


def error_breakdown(details: List[Dict]) -> Dict[str, int]:
    """Count tasks per error_type (Passed included), in a stable order."""
    counts: Dict[str, int] = {}
    for d in details:
        et = d.get("error_type") or ("Passed" if d.get("passed") else "Unknown")
        counts[et] = counts.get(et, 0) + 1
    order = {name: i for i, name in enumerate(ERROR_TYPES)}
    return dict(sorted(counts.items(), key=lambda kv: order.get(kv[0], 99)))
