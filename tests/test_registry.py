"""Tests for the versioned prompt registry: bumping and score recording."""

from __future__ import annotations

import json
from pathlib import Path

from src.registry import latest_version, record_version


def _results(accuracy=0.85):
    return {
        "n": 20,
        "accuracy": accuracy,
        "severity_accuracy": {"critical": 1.0},
        "category_accuracy": {"functional": 1.0},
        "per_severity": {"critical": 1.0},
        "per_category": {"functional": 1.0},
    }


def test_latest_version_empty_dir(tmp_path):
    assert latest_version(tmp_path / "prompts") == 0


def test_latest_version_skips_non_version_files(tmp_path):
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    (prompts / "v2.md").write_text("x")
    (prompts / "notes.md").write_text("x")
    (prompts / "v10.md").write_text("x")
    assert latest_version(prompts) == 10


def test_record_version_bumps_and_writes_files(tmp_path):
    prompts, reports = tmp_path / "prompts", tmp_path / "reports"
    v1 = record_version(prompts, reports, kind="baseline",
                        prompt_markdown="## Instruction\n\nDo the thing.",
                        results=_results(0.4), n_examples=0)
    v2 = record_version(prompts, reports, kind="optimized",
                        prompt_markdown="## Instruction\n\nDo the thing better.",
                        results=_results(0.85), n_examples=8,
                        notes="BootstrapFewShot run.")
    assert (v1, v2) == (1, 2)
    assert (prompts / "v1.md").read_text().count("40.0%") >= 1
    assert "8" in (prompts / "v2.md").read_text()

    report = (reports / "optimization_report.md").read_text()
    assert "## v1 — baseline" in report
    assert "## v2 — optimized" in report
    assert "85.0%" in report

    rows = [json.loads(line) for line in (reports / "scores.jsonl").read_text().splitlines()]
    assert [r["version"] for r in rows] == [1, 2]
    assert rows[1]["n_examples"] == 8
    assert rows[1]["accuracy"] == 0.85


def test_record_version_continues_existing_registry(tmp_path):
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    (prompts / "v3.md").write_text("existing")
    v = record_version(prompts, tmp_path / "reports", kind="optimized",
                       prompt_markdown="md", results=_results(), n_examples=4)
    assert v == 4
