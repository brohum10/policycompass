"""Compare all evidence fields; tolerate only tiny cross-platform float differences."""

import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from policycompass.synthetic import demo


def check(actual, expected, path="root"):
    if type(expected) is float:
        assert type(actual) in (int, float) and math.isclose(
            actual, expected, rel_tol=1e-9, abs_tol=1e-10
        ), path
    elif isinstance(expected, dict):
        assert isinstance(actual, dict) and actual.keys() == expected.keys(), path
        for key in expected:
            check(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected), path
        for i, (left, right) in enumerate(zip(actual, expected)):
            check(left, right, f"{path}[{i}]")
    else:
        assert type(actual) is type(expected) and actual == expected, path


if __name__ == "__main__":
    reference = Path(__file__).resolve().parents[1] / "evidence" / "synthetic.json"
    check(demo(), json.loads(reference.read_text(encoding="utf-8")))
    print("All synthetic evidence fields reproduced.")
