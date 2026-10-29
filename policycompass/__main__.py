"""Read-only command line; JSON on stdout, diagnostics on stderr."""

import argparse
import json
from pathlib import Path
import sys

from .data import Dataset, MAX_BYTES
from .estimate import compare
from .synthetic import demo, fixture


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="PolicyCompass: inspect fixed policies from bandit logs"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("demo", help="reproduce explored, thin-support, and unsupported synthetic cases")
    generator = sub.add_parser(
        "fixture", help="print a synthetic dataset with no counterfactual labels"
    )
    generator.add_argument(
        "--scenario", choices=("explored", "thin", "unsupported"), default="explored"
    )
    evaluator = sub.add_parser(
        "compare", help="compare two predeclared policies in a local JSON dataset"
    )
    evaluator.add_argument("dataset", type=Path)
    evaluator.add_argument("--baseline", default="baseline")
    evaluator.add_argument("--candidate", default="candidate")
    evaluator.add_argument("--resamples", type=int, default=400)
    evaluator.add_argument("--seed", type=int, default=20261029)
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            result = demo()
        elif args.command == "fixture":
            result = fixture(args.scenario).document
        else:
            # Bound the read, even if the file grows between opening and reading.
            with args.dataset.open("rb") as stream:
                raw = stream.read(MAX_BYTES + 1)
            if len(raw) > MAX_BYTES:
                raise ValueError("input exceeds 2 MB")
            dataset = Dataset.from_json(raw.decode("utf-8"))
            result = compare(dataset, args.baseline, args.candidate, args.resamples, args.seed)
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (ValueError, OSError, UnicodeError, RecursionError) as exc:
        print(f"policycompass: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
