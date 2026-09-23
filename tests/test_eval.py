"""Tests for the eval harness: metric correctness and result structure.

All offline: the metric is a pure function, and evaluate_program runs
against the deterministic mock LM.
"""

from __future__ import annotations

import dspy

from src.metrics import evaluate_program, exact_match


def _example(**overrides):
    base = dict(
        bug_title="t",
        bug_description="d",
        severity="high",
        category="functional",
    )
    base.update(overrides)
    return dspy.Example(**base).with_inputs("bug_title", "bug_description")


def _pred(severity, category):
    return dspy.Prediction(severity=severity, category=category)


def test_exact_match_true():
    assert exact_match(_example(), _pred("high", "functional")) == 1.0


def test_exact_match_false_on_either_field():
    assert exact_match(_example(), _pred("low", "functional")) == 0.0
    assert exact_match(_example(), _pred("high", "ui")) == 0.0
    assert exact_match(_example(), _pred("low", "ui")) == 0.0


def test_exact_match_is_case_and_space_insensitive():
    assert exact_match(_example(), _pred(" High ", "FUNCTIONAL")) == 1.0


def test_evaluate_program_structure(program, eval_examples):
    results = evaluate_program(program, eval_examples)
    assert results["n"] == len(eval_examples) == 20
    assert 0.0 <= results["accuracy"] <= 1.0
    assert set(results["per_severity"]) == {"critical", "high", "medium", "low"}
    assert set(results["per_category"]) == {
        "functional", "ui", "performance", "security", "documentation",
    }
    for mapping in (results["per_severity"], results["per_category"],
                    results["severity_accuracy"], results["category_accuracy"]):
        for value in mapping.values():
            assert 0.0 <= value <= 1.0


def test_evaluate_program_baseline_accuracy_is_mediocre(program, eval_examples):
    results = evaluate_program(program, eval_examples)
    assert 0.3 <= results["accuracy"] <= 0.6, results["accuracy"]


def test_evaluate_program_deterministic(program, eval_examples):
    first = evaluate_program(program, eval_examples)
    second = evaluate_program(program, eval_examples)
    assert first == second
