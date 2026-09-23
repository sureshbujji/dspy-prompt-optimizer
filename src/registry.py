"""Versioned prompt registry.

Every optimizer run records a new version:

    prompts/v{N}.md                  -- the prompt text for that version
    reports/optimization_report.md    -- human-readable per-version score table
    reports/scores.jsonl             -- machine-readable score log (one row/version)

Versions are monotonically increasing integers; ``latest_version()`` scans
``prompts/`` so re-running the optimizer always appends a new entry.
"""

from __future__ import annotations

import datetime as _dt
import json
import re
from pathlib import Path
from typing import Any, Dict

_VERSION_RE = re.compile(r"^v(\d+)\.md$")


def latest_version(prompts_dir: str | Path) -> int:
    """Highest recorded prompt version, or 0 when the registry is empty."""
    prompts_dir = Path(prompts_dir)
    versions = []
    if prompts_dir.is_dir():
        for p in prompts_dir.iterdir():
            m = _VERSION_RE.match(p.name)
            if m:
                versions.append(int(m.group(1)))
    return max(versions, default=0)


def record_version(
    prompts_dir: str | Path,
    reports_dir: str | Path,
    *,
    kind: str,
    prompt_markdown: str,
    results: Dict[str, Any],
    n_examples: int,
    notes: str = "",
) -> int:
    """Persist one registry entry; returns the new version number."""
    prompts_dir, reports_dir = Path(prompts_dir), Path(reports_dir)
    prompts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    version = latest_version(prompts_dir) + 1
    stamp = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")

    prompt_path = prompts_dir / f"v{version}.md"
    prompt_path.write_text(
        f"# Prompt v{version} ({kind})\n\n"
        f"- Recorded: {stamp}\n"
        f"- Held-out accuracy: {results['accuracy']:.1%} (n={results['n']})\n"
        f"- Few-shot examples in prompt: {n_examples}\n\n"
        f"{prompt_markdown.rstrip()}\n",
        encoding="utf-8",
    )

    _append_report(reports_dir / "optimization_report.md", version, kind, stamp,
                   results, n_examples, notes)

    with open(reports_dir / "scores.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "version": version,
            "kind": kind,
            "recorded": stamp,
            "n_examples": n_examples,
            "accuracy": results["accuracy"],
            "severity_accuracy": results["severity_accuracy"],
            "category_accuracy": results["category_accuracy"],
            "per_severity": results["per_severity"],
            "per_category": results["per_category"],
            "notes": notes,
        }) + "\n")

    return version


def _append_report(
    report_path: Path,
    version: int,
    kind: str,
    stamp: str,
    results: Dict[str, Any],
    n_examples: int,
    notes: str,
) -> None:
    if not report_path.exists():
        report_path.write_text(
            "# Optimization report\n\n"
            "Per-version held-out eval scores (`data/eval.jsonl`, n=20) for the\n"
            "bug-report classifier. Each entry is written by\n"
            "`python -m src.run_optimization`.\n",
            encoding="utf-8",
        )

    def _table(title: str, mapping: Dict[str, float]) -> str:
        rows = "\n".join(f"| {k} | {v:.1%} |" for k, v in mapping.items())
        return f"**{title}**\n\n| class | accuracy |\n|---|---|\n{rows}\n"

    section = (
        f"\n## v{version} — {kind} ({stamp})\n\n"
        f"- Held-out accuracy: **{results['accuracy']:.1%}** (n={results['n']})\n"
        f"- Few-shot examples in prompt: {n_examples}\n\n"
        f"{_table('Per-severity accuracy', results['per_severity'])}\n"
        f"{_table('Per-category accuracy', results['per_category'])}"
    )
    if notes:
        section += f"\n{notes}\n"
    with open(report_path, "a", encoding="utf-8") as f:
        f.write(section + "\n")
