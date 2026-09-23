"""Generate the synthetic bug-report dataset (deterministic, seed=42).

Writes data/train.jsonl (40 rows) and data/eval.jsonl (20 rows), stratified
4/2 per (severity, category) class so every class appears in both splits.

The data is synthetic on purpose: each class family shares vocabulary
(e.g. "crash/startup/launch" for critical+functional) so that a few-shot
program can plausibly learn the mapping from demonstrations while a weak
zero-shot heuristic cannot. Real bug text would be messier; this keeps the
benchmark honest about what it measures.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

CLASSES = [
    ("critical", "functional",
     ["App crashes on {p} startup", "Crash when {p} launching the app", "Data loss after {p} crash"],
     ["The app crashes {p} on startup and corrupts saved data.",
      "Immediate crash {p} during launch; unsaved work is lost.",
      "Crash {p} on cold start. Relaunch shows data loss in recent files."]),
    ("critical", "security",
     ["Authentication bypass {p} via token replay", "Unauthorized access {p} to admin panel", "Privilege escalation {p} in API"],
     ["Attackers can bypass authentication {p} by replaying a session token.",
      "Unauthorized users gain admin privileges {p} through the API.",
      "Session tokens leak {p} in redirect URLs, enabling hijack."]),
    ("high", "functional",
     ["Form submission {p} fails with validation error", "Incorrect {p} total in checkout calculation", "Export {p} fails with error"],
     ["Submitting the form {p} fails with a validation error on valid input.",
      "Checkout shows an incorrect total; the tax calculation {p} is wrong.",
      "CSV export {p} fails with a generic error after 10k rows."]),
    ("high", "security",
     ["Passwords {p} exposed in debug logs", "Session {p} never expires on mobile", "Stored XSS {p} in comment field"],
     ["Plaintext passwords appear {p} in debug logs shipped to support.",
      "Mobile sessions {p} never expire, violating the 30-minute policy.",
      "Stored XSS payload {p} executes when moderators open the queue."]),
    ("high", "performance",
     ["Dashboard {p} times out under load", "Memory leak {p} in report worker", "Search latency {p} over 8 seconds"],
     ["The dashboard {p} times out when more than 50 widgets load.",
      "Report worker leaks memory {p} until the container OOMs.",
      "Search latency {p} exceeds 8 seconds on large tenants."]),
    ("medium", "ui",
     ["Buttons {p} misaligned on settings page", "Text {p} truncated in table cells", "Overlapping {p} labels in chart legend"],
     ["Action buttons are misaligned {p} on the settings page layout.",
      "Long names get truncated {p} in table cells without a tooltip.",
      "Chart legend labels are overlapping {p} at narrow widths."]),
    ("medium", "functional",
     ["Occasional {p} glitch in list sorting", "Filter {p} needs manual refresh", "Draft {p} occasionally not saved"],
     ["List sorting has an occasional {p} glitch; a workaround is re-sorting.",
      "The filter {p} needs a manual refresh to show new items.",
      "Drafts are occasionally {p} not saved when switching tabs."]),
    ("medium", "documentation",
     ["Typo {p} in API docs quickstart", "Outdated {p} screenshots in help center", "Broken {p} link in migration guide"],
     ["A typo {p} in the API docs quickstart confuses new integrators.",
      "Help center screenshots are outdated {p} after the redesign.",
      "The migration guide has a broken {p} link to the changelog."]),
    ("low", "ui",
     ["Minor {p} spacing issue in footer", "Icon looks {p} pixelated on retina", "Tooltip {p} missing on info icon"],
     ["Footer has a minor {p} spacing inconsistency on wide screens.",
      "The status icon looks pixelated {p} on retina displays.",
      "The info icon is missing its hover {p} tooltip."]),
    ("low", "documentation",
     ["Grammar {p} issue in help article", "Inconsistent {p} capitalization in docs", "Punctuation {p} error in tooltip copy"],
     ["A help article has a grammar {p} issue in the second paragraph.",
      "Docs use inconsistent capitalization {p} for feature names.",
      "Tooltip copy has a punctuation {p} error after the update."]),
]

PHRASES = ["", "after the latest update", "on iOS", "on Android", "in dark mode",
           "for enterprise tenants", "when offline", "during peak hours", "on Safari"]


def _clean(s: str) -> str:
    return " ".join(s.split())


def main() -> None:
    rng = random.Random(42)
    items = []
    for sev, cat, titles, descs in CLASSES:
        group = []
        for i in range(6):
            title = _clean(titles[i % len(titles)].format(p=rng.choice(PHRASES)))
            desc = _clean(descs[(i + 1) % len(descs)].format(p=rng.choice(PHRASES)))
            group.append({
                "bug_title": title,
                "bug_description": desc,
                "severity": sev,
                "category": cat,
            })
        rng.shuffle(group)
        items.extend(group)

    train, evl = [], []
    for c in range(0, len(items), 6):
        group = items[c:c + 6]
        train.extend(group[:4])
        evl.extend(group[4:])
    rng.shuffle(train)
    rng.shuffle(evl)

    data_dir = Path(__file__).resolve().parent.parent / "data"
    data_dir.mkdir(exist_ok=True)
    for name, rows in (("train.jsonl", train), ("eval.jsonl", evl)):
        with open(data_dir / name, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
    print(f"wrote {data_dir/'train.jsonl'} ({len(train)} rows)")
    print(f"wrote {data_dir/'eval.jsonl'} ({len(evl)} rows)")


if __name__ == "__main__":
    main()
