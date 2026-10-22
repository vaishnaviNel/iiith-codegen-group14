"""Unit tests for the agent loops, prompts and docs checker (no model needed)."""

import unittest

from src.agent.docs_agent import check_docstring, clean_docstring, function_params, run_doc_agent
from src.agent.loop import run_agent
from src.prompts import build_code_messages, build_code_messages_reference, build_doc_messages

TESTS = ["assert add(1, 2) == 3"]
BAD = "```python\ndef add(a, b):\n    return a - b\n```"
GOOD = "```python\ndef add(a, b):\n    return a + b\n```"


class TestCodeAgent(unittest.TestCase):
    def test_fixes_after_feedback_and_sees_failure(self):
        replies, seen = iter([GOOD]), []

        def chat(msgs):
            seen.append(msgs)
            return next(replies)

        r = run_agent(build_code_messages("add", TESTS), TESTS, chat, max_attempts=3, first_output=BAD)
        self.assertTrue(r.passed)
        self.assertFalse(r.first_attempt_passed)
        self.assertEqual(r.attempts, 2)
        feedback = seen[0][-1]["content"]  # the observation the model read
        self.assertIn("failed", feedback)
        self.assertIn("assert add(1, 2) == 3", feedback)
        self.assertEqual(seen[0][-2]["content"], BAD)  # its own earlier attempt is in memory

    def test_stops_at_max_attempts(self):
        r = run_agent(build_code_messages("add", TESTS), TESTS, lambda m: BAD, max_attempts=3)
        self.assertFalse(r.passed)
        self.assertEqual(r.attempts, 3)

    def test_passing_first_output_makes_no_model_call(self):
        def chat(_):
            raise AssertionError("should not be called")

        r = run_agent([], TESTS, chat, first_output=GOOD)
        self.assertTrue(r.passed)
        self.assertEqual(r.attempts, 1)


class TestDocs(unittest.TestCase):
    CODE = "def area(width, height):\n    return width * height"

    def test_params(self):
        self.assertEqual(function_params(self.CODE), ["width", "height"])
        self.assertEqual(function_params("    def m(self, x):\n        pass"), ["x"])
        self.assertEqual(function_params("not python ("), [])

    def test_check(self):
        self.assertFalse(check_docstring("Computes it.", self.CODE)[0])
        ok, fb = check_docstring("Compute the area from width only here", self.CODE)
        self.assertFalse(ok)
        self.assertIn("height", fb)
        self.assertTrue(check_docstring("Return area given width and height.", self.CODE)[0])

    def test_clean(self):
        self.assertEqual(clean_docstring('"""Docstring: Hello there."""'), "Hello there.")
        self.assertEqual(clean_docstring("```\nSome text here\n```"), "Some text here")

    def test_doc_agent_retries(self):
        r = run_doc_agent([], self.CODE, lambda m: "Return area of width and height.", first_output="Does a thing.")
        self.assertEqual(r["attempts"], 2)
        self.assertFalse(r["first_ok"])
        self.assertTrue(r["ok"])


class TestPrompts(unittest.TestCase):
    EX = [{"payload": {"description": "d", "code": "def f(): pass", "documentation": "doc"}}]

    def test_zero_shot_has_tests_no_examples(self):
        m = build_code_messages("task", TESTS)
        self.assertEqual([x["role"] for x in m], ["system", "user"])
        self.assertIn(TESTS[0], m[-1]["content"])

    def test_turns_style_adds_pairs(self):
        self.assertEqual(len(build_code_messages("task", TESTS, self.EX)), 4)

    def test_reference_style_stays_two_messages(self):
        m = build_code_messages_reference("task", TESTS, self.EX)
        self.assertEqual(len(m), 2)
        self.assertIn("def f(): pass", m[0]["content"])

    def test_doc_prompt(self):
        self.assertIn("doc", build_doc_messages("def g(): pass", self.EX)[0]["content"])


if __name__ == "__main__":
    unittest.main()
