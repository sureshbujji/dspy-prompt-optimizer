"""Dataset loading: JSONL bug reports -> dspy.Example objects."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import dspy

INPUT_KEYS = ("bug_title", "bug_description")


def load_examples(path: str | Path) -> List[dspy.Example]:
    """Load ``{"bug_title","bug_description","severity","category"}`` rows."""
    examples = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            missing = {"bug_title", "bug_description", "severity", "category"} - row.keys()
            if missing:
                raise ValueError(f"{path}:{line_no} missing keys: {sorted(missing)}")
            examples.append(
                dspy.Example(
                    bug_title=row["bug_title"],
                    bug_description=row["bug_description"],
                    severity=str(row["severity"]).strip().lower(),
                    category=str(row["category"]).strip().lower(),
                ).with_inputs(*INPUT_KEYS)
            )
    if not examples:
        raise ValueError(f"No examples found in {path}")
    return examples
