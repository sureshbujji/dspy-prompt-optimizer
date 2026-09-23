"""Metric and evaluation harness for the bug-report classifier.

The metric is strict exact-match on both output fields -- the same function
doubles as the dspy teleprompter metric and the held-out eval scorer, so the
optimizer optimizes exactly what we report.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

import dspy

SEVERITIES = ("critical", "high", "medium", "low")
CATEGORIES = ("functional", "ui", "performance", "security", "documentation")


def _norm(value: object) -> str:
    return str(value).strip().lower()


def exact_match(example: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
    """1.0 iff both severity and category match the gold labels."""
    return float(
        _norm(pred.severity) == _norm(example.severity)
        and _norm(pred.category) == _norm(example.category)
    )


def evaluate_program(
    program: dspy.Module, devset: List[dspy.Example]
) -> Dict[str, object]:
    """Score a program on a held-out set; returns accuracy + per-class splits.

    Runs single-threaded and sequentially for determinism.
    """
    n = len(devset)
    correct = 0
    sev_correct: Dict[str, int] = defaultdict(int)
    cat_correct: Dict[str, int] = defaultdict(int)
    sev_total: Dict[str, int] = defaultdict(int)
    cat_total: Dict[str, int] = defaultdict(int)

    for ex in devset:
        pred = program(bug_title=ex.bug_title, bug_description=ex.bug_description)
        ok_sev = _norm(pred.severity) == _norm(ex.severity)
        ok_cat = _norm(pred.category) == _norm(ex.category)
        if ok_sev and ok_cat:
            correct += 1
        sev_total[_norm(ex.severity)] += 1
        cat_total[_norm(ex.category)] += 1
        sev_correct[_norm(ex.severity)] += int(ok_sev)
        cat_correct[_norm(ex.category)] += int(ok_cat)

    def rate(c: Dict[str, int], t: Dict[str, int]) -> Dict[str, float]:
        return {k: round(c[k] / t[k], 4) for k in sorted(t)}

    return {
        "n": n,
        "accuracy": round(correct / n, 4) if n else 0.0,
        "severity_accuracy": {
            k: round(sev_correct[k] / sev_total[k], 4) for k in sorted(sev_total)
        },
        "category_accuracy": {
            k: round(cat_correct[k] / cat_total[k], 4) for k in sorted(cat_total)
        },
        "per_severity": rate(sev_correct, sev_total),
        "per_category": rate(cat_correct, cat_total),
    }
