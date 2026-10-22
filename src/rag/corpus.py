"""Retrieval corpora. Each entry is {"id", "query", "payload"}:
  query   -> the text we embed and search against
  payload -> what we paste into the prompt as a few-shot example

Only TRAIN splits go in, so retrieval can never leak test answers.
"""

from __future__ import annotations

import json
from typing import Dict, List

import requests
from datasets import load_dataset

CODOC_TRAIN_URL = "https://raw.githubusercontent.com/kunpai/codocbench/main/dataset/train.jsonl"


def mbpp_query(description: str, tests: List[str] | None = None) -> str:
    """Text we embed for an MBPP task. With tests, the asserts add the function name and the
    argument/return shapes, which carry more of the task's meaning than the short description."""
    return description if not tests else description + "\n" + "\n".join(tests)


def mbpp_corpus(with_tests: bool = False) -> List[Dict]:
    """Task 1 (text -> code). Query = task description (+ tests), payload = reference solution."""
    ds = load_dataset("google-research-datasets/mbpp", "full", split="train")  # 374 tasks
    return [
        {
            "id": f"mbpp/{r['task_id']}",
            "query": mbpp_query(r["text"], r["test_list"] if with_tests else None),
            "payload": {"description": r["text"], "code": r["code"].replace("\r\n", "\n")},
        }
        for r in ds
    ]


CODOC_TEST_URL = CODOC_TRAIN_URL.replace("train.jsonl", "test.jsonl")


def codoc_test(n: int | None = None) -> List[Dict]:
    """CoDocBench TEST pairs in file order (same filter as the old pipeline, so the first n
    samples match what the CodeGen baseline sees). Never used for retrieval."""
    return codoc_corpus(CODOC_TEST_URL)[:n]


def codoc_corpus(url: str = CODOC_TRAIN_URL) -> List[Dict]:
    """Task 2 (code -> docs). Query = the function source, payload = its docstring."""
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    entries = []
    for line in resp.text.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        versions = row.get("version_data") or []
        if not versions:
            continue
        code = (versions[-1].get("code") or "").strip()
        doc = (versions[-1].get("docstring") or "").strip()
        if not code or not doc:
            continue
        project = row.get("project") or row.get("owner") or "unknown"
        func = row.get("function") or row.get("file") or "unknown"
        entries.append(
            {"id": f"{project}/{func}", "query": code, "payload": {"code": code, "documentation": doc}}
        )
    return entries
