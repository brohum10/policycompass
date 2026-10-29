"""A fixed-seed toy model router with explicit, inspectable reward probabilities."""

import random

from .data import Dataset
from .estimate import compare

SEED = 20261029
ACTIONS = ["lite", "pro", "expert"]
SEGMENTS = ["lookup", "code", "reasoning"]
REWARDS = {
    "lookup": [0.84, 0.90, 0.92],
    "code": [0.35, 0.82, 0.94],
    "reasoning": [0.20, 0.65, 0.95],
}


def choose(rng, probabilities):
    draw = rng.random()
    cumulative = 0
    for i, probability in enumerate(probabilities):
        cumulative += probability
        if draw < cumulative:
            return i
    return len(probabilities) - 1


def fixture(scenario="explored"):
    if scenario not in ("explored", "thin", "unsupported"):
        raise ValueError("unknown scenario")
    rng = random.Random(SEED)
    # The same latent contexts and rewards are used as potential outcomes in each scenario.
    rows = []
    for split, count in (("fit", 300), ("eval", 600)):
        for i in range(count):
            segment = SEGMENTS[choose(rng, [0.60, 0.25, 0.15])]
            if scenario == "explored":
                probabilities = [0.50, 0.25, 0.25]
            elif scenario == "thin":
                probabilities = [0.749, 0.25, 0.001]
            else:
                probabilities = [0.75, 0.25, 0]
            action = choose(rng, probabilities)
            uniform_reward = rng.random()
            reward = int(uniform_reward < REWARDS[segment][action])
            rows.append(
                {
                    "id": f"{split}-{i:04d}",
                    "split": split,
                    "segment": segment,
                    "action": ACTIONS[action],
                    "reward": reward,
                    "behavior": dict(zip(ACTIONS, probabilities)),
                }
            )
    baseline = {s: dict(zip(ACTIONS, [0, 1, 0])) for s in SEGMENTS}
    candidate = {s: dict(zip(ACTIONS, [1, 0, 0] if s == "lookup" else [0, 0, 1])) for s in SEGMENTS}
    return Dataset.from_document(
        {
            "version": 1,
            "actions": ACTIONS,
            "policies": {"baseline": baseline, "candidate": candidate},
            "rows": rows,
        }
    )


def truth(dataset, policy):
    """Expected synthetic reward on exactly these eval contexts; never used by estimators."""
    document = dataset.document
    values = []
    for row in document["rows"]:
        if row["split"] == "eval":
            target = document["policies"][policy][row["segment"]]
            values.append(
                sum(target[a] * REWARDS[row["segment"]][i] for i, a in enumerate(ACTIONS))
            )
    return sum(values) / len(values)


def demo():
    reports = {}
    for scenario in ("explored", "thin", "unsupported"):
        dataset = fixture(scenario)
        truth_values = {p: truth(dataset, p) for p in ("baseline", "candidate")}
        truth_values["candidate_minus_baseline"] = (
            truth_values["candidate"] - truth_values["baseline"]
        )
        try:
            report = compare(dataset)
        except ValueError as exc:
            report = {"refused": str(exc)}
        reports[scenario] = {"synthetic_truth": truth_values, "report": report}
    return {
        "seed": SEED,
        "kind": "synthetic model-router study; no provider calls",
        "reward": "binary synthetic success; costs and latency are not modeled",
        "scenarios": reports,
    }
