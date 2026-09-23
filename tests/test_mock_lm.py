"""Tests for the deterministic mock LM.

Covers: dspy LM-protocol conformance (forward() response shape), determinism,
and the weak-but-improvable behavior contract (demos lift accuracy).
"""

from __future__ import annotations

import dspy
import pytest

from src.mock_lm import WeakMockLM, _parse_fields


def test_forward_returns_openai_shaped_response(mock_lm):
    resp = mock_lm.forward(
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "[[ ## bug_title ## ]]\nT\n\n[[ ## bug_description ## ]]\nD"},
        ]
    )
    assert len(resp.choices) == 1
    choice = resp.choices[0]
    assert isinstance(choice.message.content, str)
    assert choice.finish_reason == "stop"
    assert dict(resp.usage)  # dspy calls dict(response.usage)
    assert isinstance(resp.model, str) and resp.model


def test_call_returns_parseable_completion(mock_lm):
    """dspy.__call__ -> ChatAdapter must parse severity/category out of it."""
    outputs = mock_lm(
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "[[ ## bug_title ## ]]\nApp crashes\n\n[[ ## bug_description ## ]]\nCrash on launch"},
        ]
    )
    assert isinstance(outputs, list) and len(outputs) == 1
    fields = _parse_fields(outputs[0])
    assert fields["severity"] in {"critical", "high", "medium", "low"}
    assert fields["category"] in {"functional", "ui", "performance", "security", "documentation"}


def test_deterministic_across_calls_and_instances():
    kwargs = dict(
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "[[ ## bug_title ## ]]\nSlow search\n\n[[ ## bug_description ## ]]\nSearch takes ages"},
        ]
    )
    first = WeakMockLM().forward(**kwargs).choices[0].message.content
    second = WeakMockLM().forward(**kwargs).choices[0].message.content
    third = WeakMockLM()(messages=kwargs["messages"])[0]
    assert first == second == third


def test_zero_shot_is_weak_not_perfect(program, eval_examples):
    """The baseline heuristic must be mediocre -- otherwise there is nothing to optimize."""
    from src.metrics import exact_match

    correct = sum(
        exact_match(ex, program(bug_title=ex.bug_title, bug_description=ex.bug_description))
        for ex in eval_examples
    )
    accuracy = correct / len(eval_examples)
    assert accuracy < 0.7, f"mock is too strong for a 'weak model' baseline: {accuracy}"


def test_demos_improve_accuracy(mock_lm, program, train_examples, eval_examples):
    """Few-shot demos must lift accuracy: the core simulation contract."""
    from src.metrics import evaluate_program

    baseline = evaluate_program(program, eval_examples)["accuracy"]
    compiled = dspy.LabeledFewShot(k=8).compile(program, trainset=train_examples)
    assert len(compiled.classify.demos) == 8
    optimized = evaluate_program(compiled, eval_examples)["accuracy"]
    assert optimized > baseline, f"no lift: baseline={baseline}, optimized={optimized}"


def test_seen_demo_label_is_copied(mock_lm):
    """A query identical to a demo must return that demo's labels."""
    prog = dspy.Predict("bug_title, bug_description -> severity, category")
    train = [
        dspy.Example(
            bug_title="Totally unique zebra widget",
            bug_description="The zebra widget stripes render upside down",
            severity="low",
            category="ui",
        ).with_inputs("bug_title", "bug_description")
    ]
    compiled = dspy.LabeledFewShot(k=1).compile(prog, trainset=train)
    out = compiled(
        bug_title="Totally unique zebra widget",
        bug_description="The zebra widget stripes render upside down",
    )
    assert out.severity.strip().lower() == "low"
    assert out.category.strip().lower() == "ui"


def test_is_dspy_lm_subclass():
    assert issubclass(WeakMockLM, dspy.LM)
    assert not getattr(WeakMockLM, "forward") is dspy.LM.forward
