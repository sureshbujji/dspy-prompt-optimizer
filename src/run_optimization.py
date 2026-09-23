"""Run the optimization pipeline and record a new versioned prompt entry.

Modes:
    baseline   Evaluate the zero-shot program on the held-out set and record
               it as the next prompt version (0 few-shot examples).
    optimize   Run dspy's BootstrapFewShot teleprompter on the train set,
               evaluate the compiled program on the held-out set, and record
               the optimized prompt (with its demos) as the next version.

Everything runs offline against WeakMockLM -- no API key, no network.

Usage:
    python -m src.run_optimization [--mode {baseline,optimize}]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import dspy

from src.data import load_examples
from src.metrics import evaluate_program, exact_match
from src.mock_lm import WeakMockLM
from src.program import build_program
from src.registry import record_version

ROOT = Path(__file__).resolve().parent.parent

INSTRUCTION = """Classify a software bug report by severity and business category.

Read the bug title and description, then assign exactly one severity
level and one category.

Respond with the corresponding output fields, starting with the field
`[[ ## severity ## ]]`, then `[[ ## category ## ]]`, and then ending with
the marker for `[[ ## completed ## ]]`.

- severity: one of critical, high, medium, low
- category: one of functional, ui, performance, security, documentation"""


def render_prompt_markdown(kind: str, demos: list) -> str:
    """Render the registry prompt file body for a version."""
    parts = ["## Instruction", "", INSTRUCTION, ""]
    if demos:
        parts += ["## Few-shot examples", ""]
        for i, d in enumerate(demos, 1):
            parts += [
                f"### Example {i} (label: {d.severity} / {d.category})",
                "",
                f"bug_title: {d.bug_title}",
                "",
                f"bug_description: {d.bug_description}",
                "",
                f"-> severity: {d.severity}, category: {d.category}",
                "",
            ]
    else:
        parts += ["## Few-shot examples", "", "None — zero-shot prompt.", ""]
    return "\n".join(parts)


def run_baseline(eval_set) -> tuple:
    program = build_program()
    results = evaluate_program(program, eval_set)
    return results, render_prompt_markdown("baseline", []), 0


def run_optimize(train_set, eval_set, max_bootstrapped_demos: int,
                 max_labeled_demos: int) -> tuple:
    program = build_program()
    teleprompter = dspy.BootstrapFewShot(
        metric=exact_match,
        max_bootstrapped_demos=max_bootstrapped_demos,
        max_labeled_demos=max_labeled_demos,
    )
    compiled = teleprompter.compile(program, trainset=train_set)
    results = evaluate_program(compiled, eval_set)
    demos = list(compiled.classify.demos)
    return results, render_prompt_markdown("optimized", demos), len(demos)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("baseline", "optimize"),
                        default="optimize")
    parser.add_argument("--train", default=str(ROOT / "data" / "train.jsonl"))
    parser.add_argument("--eval", default=str(ROOT / "data" / "eval.jsonl"))
    parser.add_argument("--max-bootstrapped-demos", type=int, default=4)
    # Default 12: enough to cover the class families for a strong lift, while
    # leaving headroom below the mock's saturation point (16 demos -> 100%).
    parser.add_argument("--max-labeled-demos", type=int, default=12)
    args = parser.parse_args()

    dspy.configure(lm=WeakMockLM())
    train_set = load_examples(args.train)
    eval_set = load_examples(args.eval)

    if args.mode == "baseline":
        results, prompt_md, n_examples = run_baseline(eval_set)
        notes = ("Zero-shot baseline: the mock LM answers from its weak keyword "
                 "heuristic with no demonstrations in context.")
    else:
        results, prompt_md, n_examples = run_optimize(
            train_set, eval_set, args.max_bootstrapped_demos, args.max_labeled_demos
        )
        notes = (f"BootstrapFewShot(metric=exact_match, "
                 f"max_bootstrapped_demos={args.max_bootstrapped_demos}, "
                 f"max_labeled_demos={args.max_labeled_demos}) on "
                 f"{len(train_set)} train examples.")

    version = record_version(
        ROOT / "prompts", ROOT / "reports",
        kind=args.mode, prompt_markdown=prompt_md,
        results=results, n_examples=n_examples, notes=notes,
    )

    print(f"recorded prompts/v{version}.md  ({args.mode}, {n_examples} examples)")
    print(f"held-out accuracy: {results['accuracy']:.1%} (n={results['n']})")
    print(f"per-severity: {results['per_severity']}")
    print(f"per-category: {results['per_category']}")


if __name__ == "__main__":
    main()
