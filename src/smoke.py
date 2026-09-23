"""Mock-mode smoke test: exercises the program end to end, offline.

One zero-shot prediction plus one few-shot prediction via dspy's
LabeledFewShot, all against the deterministic WeakMockLM. Exits 0 on
success, non-zero on any failure. Used by CI after pytest.

Usage:
    python -m src.smoke
"""

from __future__ import annotations

import sys
from pathlib import Path

import dspy

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import load_examples  # noqa: E402
from src.mock_lm import WeakMockLM  # noqa: E402
from src.program import build_program  # noqa: E402


def main() -> None:
    dspy.configure(lm=WeakMockLM())

    program = build_program()
    pred = program(
        bug_title="App crashes on startup",
        bug_description="Immediate crash on launch; saved data is corrupted.",
    )
    assert pred.severity.strip().lower() == "critical", pred.severity
    assert pred.category.strip().lower() == "functional", pred.category
    print(f"zero-shot smoke: {pred.severity} / {pred.category}")

    train = load_examples(ROOT / "data" / "train.jsonl")[:4]
    fewshot = dspy.LabeledFewShot(k=4).compile(build_program(), trainset=train)
    assert len(fewshot.classify.demos) == 4, "expected 4 demos attached"
    # Query identical to a demo: the mock must copy that demo's labels.
    query = train[2]
    pred2 = fewshot(bug_title=query.bug_title, bug_description=query.bug_description)
    assert pred2.severity.strip().lower() == query.severity, (pred2.severity, query.severity)
    assert pred2.category.strip().lower() == query.category, (pred2.category, query.category)
    print(f"few-shot smoke:  {pred2.severity} / {pred2.category} (copied from demo)")

    print("smoke OK: mock LM + program + teleprompter all ran offline")


if __name__ == "__main__":
    main()
