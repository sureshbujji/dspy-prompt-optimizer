# Optimization report

Per-version held-out eval scores (`data/eval.jsonl`, n=20) for the
bug-report classifier. Each entry is written by
`python -m src.run_optimization`.

## v1 — baseline (2026-09-23 04:27)

- Held-out accuracy: **50.0%** (n=20)
- Few-shot examples in prompt: 0

**Per-severity accuracy**

| class | accuracy |
|---|---|
| critical | 50.0% |
| high | 33.3% |
| low | 0.0% |
| medium | 100.0% |

**Per-category accuracy**

| class | accuracy |
|---|---|
| documentation | 100.0% |
| functional | 100.0% |
| performance | 100.0% |
| security | 100.0% |
| ui | 50.0% |

Zero-shot baseline: the mock LM answers from its weak keyword heuristic with no demonstrations in context.


## v2 — optimize (2026-09-23 04:27)

- Held-out accuracy: **85.0%** (n=20)
- Few-shot examples in prompt: 12

**Per-severity accuracy**

| class | accuracy |
|---|---|
| critical | 75.0% |
| high | 100.0% |
| low | 100.0% |
| medium | 66.7% |

**Per-category accuracy**

| class | accuracy |
|---|---|
| documentation | 100.0% |
| functional | 100.0% |
| performance | 100.0% |
| security | 100.0% |
| ui | 50.0% |

BootstrapFewShot(metric=exact_match, max_bootstrapped_demos=4, max_labeled_demos=12) on 40 train examples.

