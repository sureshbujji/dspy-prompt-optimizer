"""The dspy program under optimization: a bug-report classifier."""

from __future__ import annotations

import dspy


class ClassifyBugReport(dspy.Signature):
    """Classify a software bug report by severity and business category.

    Read the bug title and description, then assign exactly one severity
    level and one category.
    """

    bug_title: str = dspy.InputField(desc="Short title of the bug report.")
    bug_description: str = dspy.InputField(
        desc="Longer free-text description of the bug: repro steps, impact, environment."
    )
    severity: str = dspy.OutputField(desc="One of: critical, high, medium, low.")
    category: str = dspy.OutputField(
        desc="One of: functional, ui, performance, security, documentation."
    )


class BugReportClassifier(dspy.Module):
    """Thin module wrapping a single Predict step over ClassifyBugReport.

    Keeping the module to one Predict call is deliberate: it isolates the
    effect of prompt optimization (the teleprompter only has one predictor's
    demos/instructions to tune), so the accuracy lift is attributable to the
    optimizer, not to program structure.
    """

    def __init__(self) -> None:
        super().__init__()
        self.classify = dspy.Predict(ClassifyBugReport)

    def forward(self, bug_title: str, bug_description: str) -> dspy.Prediction:
        return self.classify(bug_title=bug_title, bug_description=bug_description)


def build_program() -> BugReportClassifier:
    return BugReportClassifier()
