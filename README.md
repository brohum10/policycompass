# PolicyCompass

**Would a different AI router actually work better—or do your old logs just make it look that way?**

[![Checks](https://github.com/brohum10/policycompass/actions/workflows/ci.yml/badge.svg)](https://github.com/brohum10/policycompass/actions/workflows/ci.yml)

An offline, inspectable **contextual-bandit policy evaluation lab**. Compare two fixed routing policies when each historical request reveals the outcome of only the action that was taken. Inspect four estimators, paired uncertainty intervals, segment breakdowns, and evidence-quality warnings before drawing conclusions.

Python 3.12+ · standard library only · no API keys · no provider calls · local CLI and library

## Run it

From the downloaded repository folder:

```sh
python -m policycompass demo
python -m unittest discover -s tests -v
python scripts/verify_reference.py
```

The demo has three reproducible cases: an explored router, a router that almost never tries its expert action, and a router with no support for that action. It prints JSON and never changes files or calls an AI service.

To inspect the exact input and evaluate it separately:

```sh
python -m policycompass fixture > router.local.json
python -m policycompass compare router.local.json --resamples 400
python -m policycompass fixture --scenario unsupported > unsupported.local.json
python -m policycompass compare unsupported.local.json
```

The last command exits **2** with an unsupported-action error and no partial report. A successful comparison exits **0**, including when warnings are present. This is a diagnostic tool, **not a release gate**. Shell redirection can overwrite files; choose fresh output names. `*.local.json` and `private-data/` are ignored by Git.

Optional installation exposes the `policycompass` command:

```sh
python -m pip install .
policycompass --help
```

Installing can download build tooling; running from the repository needs no third-party packages or network access.

## A real problem, demonstrated with synthetic data

Imagine a router choosing `lite`, `pro`, or `expert` for `lookup`, `code`, and `reasoning` requests. Your logs contain the chosen action, its success, and its actual selection probability. They do **not** contain the outcome of the other two actions. Comparing raw averages of different actions confuses routing choices with request difficulty.

Here, the baseline always chooses `pro`. The candidate chooses `lite` for lookup and `expert` for the other segments. Neither policy is learned or selected using the evaluation rewards. The reward model is a small empirical table fitted on a separate partition.

The generator uses **300 fit rows and 600 eval rows**, seed `20261029`, and explicit Bernoulli success probabilities. The generator's counterfactual probability table is only used to report synthetic truth; it is absent from estimator inputs. There are no real models, customers, costs, or latency measurements.

### What the fixed example actually shows

Values are estimated expected success rewards. These are **not production improvements or accuracy benchmarks**.

| Explored logging: 600 eval rows | Baseline | Candidate | Candidate − baseline |
| --- | ---: | ---: | ---: |
| Known synthetic expectation on these contexts | 0.8371 | 0.8842 | +0.0471 |
| Direct method (DM) | 0.8295 | 0.8881 | +0.0586 |
| Inverse propensity scoring (IPS) | 0.9067 | 0.8233 | −0.0833 |
| Self-normalized IPS (SNIPS) | 0.8193 | 0.8821 | +0.0629 |
| Doubly robust (DR) | 0.8306 | 0.8866 | +0.0561 |

The DR difference's approximate paired percentile interval is **[−0.0150, +0.1374]**. This sample does **not** establish a positive gain. IPS even reverses the ordering in this finite sample; the example keeps that result visible rather than choosing the prettiest estimator.

| Candidate diagnostics | Explored | Thin exploration |
| --- | ---: | ---: |
| Expert logging probability | 0.25 | 0.001 |
| Largest observed importance weight | 4 | 1,000 |
| Effective sample size diagnostic | 200.00 | 1.80 |
| DR estimate | 0.8866 | 1.3078 |
| Mean target mass using reward-model fallback | 0% | 42.5% |

A reward estimate above 1 is possible for **unclipped IPS/DR**; it signals estimator instability, not superhuman success. Thin exploration retains the same synthetic truth but makes the candidate DR difference interval roughly **[−0.0964, +1.5811]**. With expert probability zero, evaluation is refused entirely. Positive logging probability alone does not mean the finite sample is useful.

Every report field is checked against [the committed evidence](evidence/synthetic.json), including failed support, model cells, hashes, subgroup summaries, warnings, and intervals. No speed or memory benchmark is claimed.

## How it works

```text
                    strict JSON + probability/support validation
                                      │
          ┌───────────────────────────┴───────────────────────────┐
          │                                                       │
    independent fit rows                                    held-out eval rows
    empirical reward table                                  logged action + reward
    + global-mean fallback                                  full logging distribution
          │                                                       │
          └────────────── fixed baseline / candidate ──────────────┘
                                      │
                       per-row DM / IPS / DR scores
                                      │
                  averages + SNIPS ratio + segment diagnostics
                                      │
                   paired row bootstrap; fixed reward model
                                      │
                     JSON evidence, warnings, no deployment
```

For context `x`, logged action `a`, reward `r`, logging policy `b`, target policy `π`, and independent reward estimate `q`:

```text
w    = π(a | x) / b(a | x)
DM   = mean( sum_a π(a | x) q(x, a) )
IPS  = mean( w r )
SNIPS= sum(w r) / sum(w)
DR   = mean( sum_a π(a | x) q(x, a) + w [r − q(x, a)] )
ESS  = sum(w)² / sum(w²)
```

- **DM** predicts outcomes from the fit partition. Missing segment/action cells use the global fit reward mean; the target probability mass depending on that fallback is reported. A narrow DM interval can hide a wrong reward model.
- **IPS** corrects for the old router's action probabilities. Very rare actions can produce enormous weights and variance.
- **SNIPS** normalizes the weights; it is generally biased in finite samples. When their sum is zero, its value is `null`, never zero.
- **DR** combines a reward prediction and an importance-weighted residual. Under the appropriate identification/support and sampling assumptions, correct propensities or a correct reward model can protect the value estimate. It does not eliminate finite-sample variance or validate your logs.
- **ESS** is a weight-concentration diagnostic, not a literal count of independent rows or a guarantee of interval quality. We also report mean weight, possible and observed maximum weights, positive-weight rows, and target cells with no eval observations.

The reward model is intentionally transparent: mean reward for each observed segment/action pair. It does not predict from embeddings, train an LLM, or optimize a policy. The fit hash binds the normalized fit rows; modifying evaluation rewards cannot change the model.

## Input contract

Top level: exactly `version`, `actions`, `policies`, and `rows`.

| Field | Meaning |
| --- | --- |
| `version` | Integer `1` |
| `actions` | 2–8 distinct action names |
| `policies` | 2–8 named, fixed policies; each maps every segment to a full action-probability object |
| `rows[].id` | Globally unique row identifier across both partitions |
| `rows[].split` | `fit` or `eval`; at least 1 fit row and 2 eval rows |
| `rows[].segment` | The full categorical context used by this lab's policy/model |
| `rows[].action` | The action actually selected |
| `rows[].reward` | The observed reward, a finite number in `[0, 1]` |
| `rows[].behavior` | Full distribution used to sample that action **at that decision** |

Names use 1–80 ASCII letters, digits, or `_.:-`. Limits: 30 segments, 2,000 total rows, and 2 MB input. Unknown/duplicate JSON fields, duplicate row IDs, booleans as numbers, non-finite rewards, invalid sums, and zero probability for the logged action are rejected. Nonzero probabilities must be at least `1e-6`; near-unit sums within `1e-12` are normalized. Actions and rows are sorted for reproducible snapshots and hashes.

The behavior distribution may differ on every row. Support checks inspect **all target actions on every eval context**, not just the action observed in that row. A target probability above zero with behavior probability zero is an error. No hidden flooring, clipping, or pooled fallback repairs missing support.

Meaningful counterfactual interpretation requires actual randomized selection with correctly logged probabilities, stable reward definitions/processes, adequate support, independent fit/eval data, and policies fixed before examining eval rewards. Treat each segment as the context available to this lab's policies; arbitrary hidden selection bias is not corrected by supplying a plausible probability object. IID evaluation rows are additionally needed for this bootstrap workflow.

The schema can verify the shape of a logged probability, **not whether it is truthful**. Do not reconstruct probabilities from action frequencies or simply invent an exploration rate for deterministic logs.

## Uncertainty and warnings

The same sampled eval-row indices are used for both policies, preserving their pairing. Each resample recomputes all four estimates, including the SNIPS denominator. The model remains fixed. The report uses linearly interpolated 2.5th/97.5th percentiles of 400 resamples by default, configurable from 50–2,000 with an explicit seed.

This is an **approximate IID row bootstrap**, conditional on the fitted reward model. It does not include reward-model training uncertainty and is not a finite-sample confidence guarantee. Repeated users, sessions, adaptive logging, nonstationarity, selection of policies after inspecting eval outcomes, or extremely heavy weights need different study designs. Synthetic truth here is conditional on the exact demo contexts; bootstrap intervals also vary context composition and are not matched conditional-coverage guarantees.

If **any** resample has an undefined estimate, that method's interval is `null` and its undefined-resample count is retained. Dropping those draws would hide a support problem. Even a non-null interval can be misleading when rare target actions were never observed; check the missing-cell and possible-weight diagnostics.

Warnings use transparent heuristics: ESS below 10% of eval rows, possible weight above 10, any missing target cell, reward-model fallback, or any undefined bootstrap estimate. They are not statistically calibrated thresholds. Segment summaries contain point estimates and weight diagnostics; only overall estimates/differences receive bootstrap intervals. Output has no automatic winner or deploy decision.

## Library example

```python
from pathlib import Path
from policycompass import Dataset, compare

dataset = Dataset.from_json(Path("router.local.json").read_text(encoding="utf-8"))
report = compare(dataset, baseline="baseline", candidate="candidate", resamples=400)
print(report["candidate_minus_baseline"]["dr"])
print(report["policies"]["candidate"]["warnings"])
```

Construct datasets through `from_json` or `from_document`; direct dataclass construction and low-level score helpers assume trusted library callers. The CLI is the validated input boundary. The snapshot is immutable; `document` returns a detached copy. Hashes identify reproducible input/model snapshots; they do not authenticate who produced the data.

## Tests and boundaries

Tests include hand-calculated scores, exhaustive action enumeration against known expectations across 80 seeded parameter sets, a four-history IPS expectation check, held-out-label isolation, identical-policy paired intervals, absent-action support checks, missing-model cells, undefined bootstrap resamples, no hidden clipping, malformed inputs, and CLI read-only behavior. CI checks Python 3.12/3.14 on Linux and Windows, replays all committed evidence, and installs the package.

This is a bounded engineering/ML learning project developed with AI assistance. The algorithms implement established methods; the contribution is the inspectable workflow, explicit failure cases, tests, and evidence. It is not a novel estimator, live router, sequential-RL evaluator, causal-identification guarantee, privacy system, or replacement for a randomized online experiment. It cannot recover unobserved counterfactual outcomes from unsupported data. Only one-step rewards are modeled; there are no costs, delayed effects, resource constraints, external integrations, or policy-training loop.

Method references: [Dudík, Langford & Li, *Doubly Robust Policy Evaluation and Learning* (ICML 2011)](https://www.microsoft.com/en-us/research/publication/doubly-robust-policy-evaluation-and-learning-2/) and [Vowpal Wabbit's off-policy evaluation tutorial](https://vowpalwabbit.org/tutorials/off_policy_evaluation.html). These motivate the estimators; this project implements a small, independent tabular version.

MIT license. See [LICENSE](LICENSE). See [SECURITY.md](SECURITY.md) before using private logs.
