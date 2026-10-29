"""Auditable one-step estimators. Policies and reward model stay fixed during resampling."""

from dataclasses import dataclass
import math
import random

from .data import Dataset, digest

METHODS = ("dm", "ips", "snips", "dr")


@dataclass(frozen=True)
class RewardModel:
    cells: tuple
    fallback: float
    training_sha256: str

    def predict(self, segment, action):
        for group, arm, mean, count in self.cells:
            if (group, arm) == (segment, action):
                return mean
        return self.fallback

    def observed(self, segment, action):
        return any((s, a) == (segment, action) for s, a, _, _ in self.cells)

    def document(self):
        return {
            "training_sha256": self.training_sha256,
            "fallback": self.fallback,
            "cells": [
                {"segment": s, "action": a, "mean": m, "rows": n} for s, a, m, n in self.cells
            ],
        }


def fit(dataset):
    """Only fit rows influence the empirical reward table and global-mean fallback."""
    document = dataset.document
    rows = [row for row in document["rows"] if row["split"] == "fit"]
    buckets = {}
    for row in rows:
        buckets.setdefault((row["segment"], row["action"]), []).append(row["reward"])
    cells = tuple(
        (s, a, math.fsum(values) / len(values), len(values))
        for (s, a), values in sorted(buckets.items())
    )
    return RewardModel(cells, math.fsum(row["reward"] for row in rows) / len(rows), digest(rows))


def contributions(dataset, model, policy):
    """Return one score tuple per eval row: weight, weighted reward, DM, DR."""
    document = dataset.document
    if policy not in document["policies"]:
        raise ValueError(f"unknown policy: {policy}")
    mapping = document["policies"][policy]
    scores, fallback_mass = [], []
    for row in document["rows"]:
        if row["split"] != "eval":
            continue
        target = mapping[row["segment"]]
        # Check every action, including actions absent from this particular logged sample.
        for action, probability in target.items():
            if probability > 0 and row["behavior"][action] == 0:
                raise ValueError(f"unsupported policy {policy} at row {row['id']}: {action}")
        q = {a: model.predict(row["segment"], a) for a in document["actions"]}
        dm = math.fsum(target[a] * q[a] for a in q)
        weight = target[row["action"]] / row["behavior"][row["action"]]
        ips = weight * row["reward"]
        dr = dm + weight * (row["reward"] - q[row["action"]])
        scores.append((weight, ips, dm, dr))
        fallback_mass.append(
            math.fsum(p for a, p in target.items() if not model.observed(row["segment"], a))
        )
    return tuple(scores), math.fsum(fallback_mass) / len(fallback_mass)


def summarize(scores):
    """No clipping: IPS and DR may leave [0, 1]; zero-weight SNIPS is undefined."""
    if not scores:
        raise ValueError("at least one score is required")
    n = len(scores)
    weight = math.fsum(row[0] for row in scores)
    weighted_reward = math.fsum(row[1] for row in scores)
    squared = math.fsum(row[0] ** 2 for row in scores)
    return {
        "estimates": {
            "dm": math.fsum(row[2] for row in scores) / n,
            "ips": weighted_reward / n,
            "snips": weighted_reward / weight if weight else None,
            "dr": math.fsum(row[3] for row in scores) / n,
        },
        "diagnostics": {
            "rows": n,
            "positive_weight_rows": sum(row[0] > 0 for row in scores),
            "weight_sum": weight,
            "mean_weight": weight / n,
            "max_weight": max(row[0] for row in scores),
            "effective_sample_size": weight**2 / squared if squared else 0.0,
        },
    }


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def _interval(values, total):
    undefined = sum(value is None for value in values)
    # Dropping undefined resamples would hide poor empirical support.
    bounds = None if undefined else [percentile(values, 0.025), percentile(values, 0.975)]
    return {"percentile_95": bounds, "undefined_resamples": undefined, "resamples": total}


def compare(dataset, baseline="baseline", candidate="candidate", resamples=400, seed=20261029):
    if not isinstance(dataset, Dataset):
        raise ValueError("provide a validated Dataset")
    if baseline == candidate:
        raise ValueError("baseline and candidate must name different fixed policies")
    if type(resamples) is not int or not 50 <= resamples <= 2000:
        raise ValueError("resamples must be integer within [50, 2000]")
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("seed must be an integer within [0, 2**64)")
    model = fit(dataset)
    policies, score_sets = {}, []
    for policy in (baseline, candidate):
        scores, fallback_mass = contributions(dataset, model, policy)
        report = summarize(scores)
        eval_rows = [r for r in dataset.document["rows"] if r["split"] == "eval"]
        report["by_segment"] = {
            segment: summarize(
                [score for row, score in zip(eval_rows, scores) if row["segment"] == segment]
            )
            for segment in sorted({row["segment"] for row in eval_rows})
        }
        d = report["diagnostics"]
        d["mean_target_mass_using_fallback"] = fallback_mass
        mapping = dataset.document["policies"][policy]
        d["max_possible_weight_on_eval_contexts"] = max(
            p / row["behavior"][action]
            for row in eval_rows
            for action, p in mapping[row["segment"]].items()
            if p > 0
        )
        observed = {(r["segment"], r["action"]) for r in eval_rows}
        needed = {
            (r["segment"], a) for r in eval_rows for a, p in mapping[r["segment"]].items() if p > 0
        }
        d["target_cells_without_eval_observations"] = [
            {"segment": s, "action": a} for s, a in sorted(needed - observed)
        ]
        warnings = []
        if d["effective_sample_size"] < 0.1 * d["rows"]:
            warnings.append("low_effective_sample_size")
        if d["max_possible_weight_on_eval_contexts"] > 10:
            warnings.append("large_importance_weight")
        if d["target_cells_without_eval_observations"]:
            warnings.append("target_cell_not_observed_in_eval")
        if fallback_mass > 0:
            warnings.append("reward_model_fallback_used")
        report["warnings"] = warnings
        policies[policy] = report
        score_sets.append(scores)
    n = len(score_sets[0])
    rng = random.Random(seed)
    draws = {p: {method: [] for method in METHODS} for p in (baseline, candidate)}
    deltas = {method: [] for method in METHODS}
    for _ in range(resamples):
        # Identical indices preserve pairing between the two policies.
        indices = [rng.randrange(n) for _ in range(n)]
        estimates = [summarize([scores[i] for i in indices])["estimates"] for scores in score_sets]
        for method in METHODS:
            left, right = (estimate[method] for estimate in estimates)
            draws[baseline][method].append(left)
            draws[candidate][method].append(right)
            deltas[method].append(None if left is None or right is None else right - left)
    difference = {}
    for method in METHODS:
        left, right = (policies[p]["estimates"][method] for p in (baseline, candidate))
        difference[method] = {
            "estimate": None if left is None or right is None else right - left,
            **_interval(deltas[method], resamples),
        }
    for policy in (baseline, candidate):
        policies[policy]["intervals"] = {
            m: _interval(v, resamples) for m, v in draws[policy].items()
        }
        if any(item["undefined_resamples"] for item in policies[policy]["intervals"].values()):
            policies[policy]["warnings"].append("undefined_bootstrap_estimate")
    return {
        "version": 1,
        "dataset_sha256": dataset.sha256,
        "reward_model": model.document(),
        "baseline": baseline,
        "candidate": candidate,
        "policies": policies,
        "candidate_minus_baseline": difference,
        "bootstrap": {
            "seed": seed,
            "resamples": resamples,
            "unit": "independent eval row",
            "model": "fixed; fit rows are not resampled",
        },
        "interpretation": "Diagnostic estimates only; not an automatic deployment decision.",
    }
