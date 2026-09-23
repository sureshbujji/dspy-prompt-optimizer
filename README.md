# dspy-prompt-optimizer

I built this to answer a question I kept running into as a QA lead evaluating AI features: **when a prompt change "improves" a model, how do you prove it with numbers instead of vibes?**

This repo is a metric-driven prompt optimization loop built on the real `dspy` package (2.6.24). The task is a bug-report classifier — bug title + description → severity + category — expressed as a typed `dspy.Signature` inside a `dspy.Module`. A deterministic mock LM stands in for a real model so the whole thing runs offline, and dspy's `BootstrapFewShot` teleprompter optimizes the prompt against an exact-match metric. Every run is scored on a held-out set and recorded in a versioned prompt registry.

Held-out result on this repo's data: **50.0% → 85.0%** (v1 zero-shot → v2 optimized, n=20).

## Architecture

```
                    +-------------------+
                    |  bug title +      |
                    |  bug description  |
                    +--------+----------+
                             |
              +--------------v---------------+
              |  BugReportClassifier         |  src/program.py
              |  (dspy.Module: one Predict  |
              |   over ClassifyBugReport     |
              |   signature)                 |
              +--------------+----------------+
                             |  ChatAdapter renders
                             |  [[ ## field ## ]] messages
              +--------------v---------------+
              |  WeakMockLM                  |  src/mock_lm.py
              |  (dspy.LM subclass,          |
              |   forward() only;            |
              |   weak heuristic zero-shot,  |
              |   demo-matching few-shot)    |
              +--------------+----------------+
                             |
        +--------------------+--------------------+
        |                                         |
 +------v------+                          +-------v-------+
 | exact_match |                          | BootstrapFewShot|  src/run_optimization.py
 | metric      |  src/metrics.py          | (teleprompter:   |
 | (severity & |                          |  4 bootstrapped |
 |  category)  |                          |  + 8 labeled    |
 +------+------+                          |  demos)         |
        |                                 +-------+-------+
        |                                         |
        +-------------+---------------------------+
                      |
             +--------v---------+
             | held-out eval    |  data/eval.jsonl (n=20)
             | per-class scores |
             +--------+---------+
                      |
             +--------v--------------------------+
             | prompt registry                   |
             |  prompts/v1.md, v2.md, ...        |  src/registry.py
             |  reports/optimization_report.md   |
             |  reports/scores.jsonl             |
             +-----------------------------------+
```

One `Predict` call, one metric, one teleprompter — deliberately minimal, so the accuracy lift is attributable to prompt optimization and nothing else.

## The honest part: what the mock is and isn't

dspy needs an LM, and I didn't want an API key or network in this repo. So `src/mock_lm.py` implements a **deterministic mock** that subclasses the real `dspy.LM` and overrides only `forward()` — the one method dspy's `BaseLM` requires (`dspy/clients/base_lm.py` in dspy 2.6.24):

```python
forward(self, prompt=None, messages=None, **kwargs) -> response
```

where `response` must look like an OpenAI response object:

| attribute | what dspy does with it |
|---|---|
| `response.choices` | list; dspy reads `choice.message.content` (str) and `choice.finish_reason` |
| `response.usage` | dspy calls `dict(response.usage)` for history logging |
| `response.model` | str, echoed into LM history |

The mock returns `[[ ## severity ## ]]` / `[[ ## category ## ]]` blocks, which dspy's `ChatAdapter` parses back into the signature's output fields — the exact same path a real LM response takes.

Behavior contract (documented, tested, and the whole point of the simulation):

- **Zero-shot** (no demos in the prompt): a deliberately underfit keyword heuristic — `crash`/`data loss` → critical, `error` → high, else medium; category from a small keyword list. Scores **50%** on the held-out set.
- **Few-shot** (demos present): the mock pairs each assistant demo message with its user message, scores demos by content-word overlap with the query, and returns the winning demo's labels (ties → earliest demo; zero overlap → weak heuristic). With 12 demos this scores **85%**.

This *simulates* in-context learning; it is not the real thing. A real LM generalizes from demos through its weights; this mock does nearest-demo keyword matching, deterministically. What *is* real: the dspy machinery — signatures, adapters, teleprompters, metrics, and the eval loop — all running end to end, which is what this repo demonstrates. Swap in a real LM (below) and the same pipeline optimizes against real model behavior.

## Which optimizer ran, and why

`dspy.BootstrapFewShot(metric=exact_match, max_bootstrapped_demos=4, max_labeled_demos=12)` — the standard dspy few-shot optimizer. It runs the program (as its own teacher) over the 40 train examples, keeps the traces the metric scores as correct (4 bootstrapped demos), and fills the prompt out to 12 demos with labeled train examples (dspy 2.6.x caps the total at `max_labeled_demos`; sampling is seeded, so runs are reproducible). I verified it runs fully offline against the mock. `LabeledFewShot` is used in the smoke test and unit tests as the lighter-weight alternative; both work, and the README records exactly which one produced each artifact.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # dspy==2.6.24, pytest==9.1.1

python -m pytest -q               # 17 tests, fully offline
python -m src.smoke               # mock-mode smoke test (also runs in CI)

python -m src.run_optimization --mode baseline   # score zero-shot, record next version
python -m src.run_optimization --mode optimize   # BootstrapFewShot, score, record next version
python scripts/generate_data.py   # regenerate data/train.jsonl + data/eval.jsonl (seed=42)
```

No `.env`, no keys, no network needed. (`.env.example` documents the optional `OPENAI_API_KEY`.)

## Sample output

```
$ python -m src.run_optimization --mode baseline
recorded prompts/v1.md  (baseline, 0 examples)
held-out accuracy: 50.0% (n=20)
per-severity: {'critical': 0.5, 'high': 0.3333, 'low': 0.0, 'medium': 1.0}
per-category: {'documentation': 1.0, 'functional': 1.0, 'performance': 1.0, 'security': 1.0, 'ui': 0.5}

$ python -m src.run_optimization --mode optimize
Bootstrapped 4 full traces after 10 examples for up to 1 rounds, amounting to 10 attempts.
recorded prompts/v2.md  (optimize, 12 examples)
held-out accuracy: 85.0% (n=20)
per-severity: {'critical': 0.75, 'high': 1.0, 'low': 1.0, 'medium': 0.6667}
per-category: {'documentation': 1.0, 'functional': 1.0, 'performance': 1.0, 'security': 1.0, 'ui': 0.5}
```

`reports/optimization_report.md` accumulates one scored section per version (accuracy, #examples, per-severity and per-category tables); `reports/scores.jsonl` keeps the machine-readable log. Re-running the optimizer appends v3, v4, … — deterministically, so v3 reproduces v2's demos.

## Layout

```
src/
  mock_lm.py            # WeakMockLM: dspy.LM subclass, deterministic, offline
  program.py            # ClassifyBugReport signature + BugReportClassifier module
  data.py               # JSONL -> dspy.Example loader
  metrics.py            # exact_match metric + held-out eval harness
  registry.py           # version bumping, prompt files, report + scores log
  run_optimization.py   # CLI: baseline / optimize, records a new registry version
  smoke.py              # CI smoke test (zero-shot + few-shot, offline)
scripts/generate_data.py# synthetic dataset generator (seed=42) -> data/*.jsonl
data/                   # train.jsonl (40), eval.jsonl (20), stratified by class
prompts/                # v1.md (baseline), v2.md (optimized, 12 demos)
reports/                # optimization_report.md, scores.jsonl
tests/                  # mock determinism + LM protocol, eval harness, registry
```

## Swapping to a real LM (one line)

In `src/run_optimization.py`, replace:

```python
dspy.configure(lm=WeakMockLM())
```

with:

```python
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini", api_key=os.environ["OPENAI_API_KEY"]))
```

Everything downstream — program, metric, teleprompter, registry — is LM-agnostic and works unchanged. Expect the numbers to move: the mock's demo-matching saturates the synthetic set at 16 demos, while a real LM will show a noisier, more interesting optimization curve.

## Roadmap

- **Instruction optimization**: add `COPRO`/`MIPROv2` runs and record them as v3+ to compare instruction-tuning vs few-shot demos on the same metric.
- **Harder eval set**: adversarial bug reports with misleading keywords (e.g. "punctuation error" labeled low/documentation) to stress-test demo selection.
- **Real-LM validation**: run the identical pipeline against `gpt-4o-mini` and publish the mock-vs-real accuracy delta as a calibration note.
- **Cost tracking**: log tokens/LM calls per optimizer run into `scores.jsonl` for cost-per-point-of-accuracy analysis.
- **Multi-step program**: extend the module to retrieve-then-classify (bug dedup lookup before classification) and optimize the full program with `BootstrapFewShotWithRandomSearch`.
