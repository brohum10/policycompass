"""Strict, detached input snapshots; logging probabilities are supplied, never guessed."""

from dataclasses import dataclass
import hashlib
import json
import math
import re

MAX_BYTES = 2_000_000
NAME = re.compile(r"[A-Za-z0-9_.:-]{1,80}\Z")


def name(value):
    if not isinstance(value, str) or not NAME.fullmatch(value):
        raise ValueError("names must be 1–80 ASCII letters, digits, or _.:-")
    return value


def keys(value, expected, label):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f"{label} requires exactly {sorted(expected)}")


def distribution(value, actions):
    keys(value, actions, "distribution")
    for probability in value.values():
        if type(probability) not in (int, float):
            raise ValueError("probabilities must be finite numbers")
        if not 0 <= probability <= 1 or 0 < probability < 1e-6 or not math.isfinite(probability):
            raise ValueError("probabilities must be zero or within [1e-6, 1]")
    total = math.fsum(value.values())
    if not math.isclose(total, 1.0, rel_tol=0, abs_tol=1e-12):
        raise ValueError("probabilities must sum to one")
    return {action: float(value[action]) / total for action in actions}


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


@dataclass(frozen=True)
class Dataset:
    """Construct via from_document/from_json. document returns a fresh copy each time."""

    encoded: str

    @classmethod
    def from_json(cls, raw):
        if not isinstance(raw, str) or len(raw.encode("utf-8")) > MAX_BYTES:
            raise ValueError("input must be UTF-8 JSON no larger than 2 MB")
        try:
            document = json.loads(raw, object_pairs_hook=unique_object)
        except (RecursionError, json.JSONDecodeError) as exc:
            raise ValueError("invalid or excessively nested JSON") from exc
        return cls.from_document(document)

    @classmethod
    def from_document(cls, document):
        keys(document, ("version", "actions", "policies", "rows"), "dataset")
        if type(document["version"]) is not int or document["version"] != 1:
            raise ValueError("version must be integer 1")
        actions = document["actions"]
        if not isinstance(actions, list) or not 2 <= len(actions) <= 8:
            raise ValueError("provide 2–8 actions")
        actions = sorted(name(action) for action in actions)
        if len(set(actions)) != len(actions):
            raise ValueError("duplicate action")
        rows = document["rows"]
        if not isinstance(rows, list) or not 3 <= len(rows) <= 2000:
            raise ValueError("provide 3–2000 rows")
        normalized, ids, segments = [], set(), set()
        for row in rows:
            keys(row, ("id", "split", "segment", "action", "reward", "behavior"), "row")
            row_id, segment, action = name(row["id"]), name(row["segment"]), name(row["action"])
            if row_id in ids:
                raise ValueError("row IDs must be globally unique across splits")
            ids.add(row_id)
            segments.add(segment)
            if row["split"] not in ("fit", "eval") or action not in actions:
                raise ValueError("invalid split or unknown logged action")
            reward = row["reward"]
            if (
                type(reward) not in (int, float)
                or not 0 <= reward <= 1
                or not math.isfinite(reward)
            ):
                raise ValueError("reward must be a finite number within [0, 1]")
            behavior = distribution(row["behavior"], actions)
            if behavior[action] == 0:
                raise ValueError("logged action cannot have zero logging probability")
            normalized.append(dict(row, reward=float(reward), behavior=behavior))
        if len(segments) > 30:
            raise ValueError("at most 30 segments are supported")
        if sum(row["split"] == "fit" for row in normalized) < 1:
            raise ValueError("at least one independent fit row is required")
        if sum(row["split"] == "eval" for row in normalized) < 2:
            raise ValueError("at least two eval rows are required")
        policies = document["policies"]
        if not isinstance(policies, dict) or not 2 <= len(policies) <= 8:
            raise ValueError("provide 2–8 fixed policies")
        cleaned = {}
        for policy, mapping in policies.items():
            name(policy)
            keys(mapping, segments, "policy segments")
            cleaned[policy] = {
                segment: distribution(mapping[segment], actions) for segment in sorted(segments)
            }
        value = {
            "version": 1,
            "actions": actions,
            "policies": cleaned,
            "rows": sorted(normalized, key=lambda row: row["id"]),
        }
        encoded = canonical(value)
        if len(encoded.encode()) > MAX_BYTES:
            raise ValueError("normalized dataset exceeds 2 MB")
        return cls(encoded)

    @property
    def document(self):
        return json.loads(self.encoded)

    @property
    def sha256(self):
        return hashlib.sha256(self.encoded.encode()).hexdigest()
