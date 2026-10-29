import copy
import io
import itertools
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

from policycompass import Dataset, compare, contributions, fit, summarize
from policycompass.__main__ import main
from policycompass.data import distribution
from policycompass.estimate import RewardModel, percentile
from policycompass.synthetic import fixture, truth


def small():
    rows = []
    for row_id, split, action, reward in (
        ("f1", "fit", "a", 0.2),
        ("f2", "fit", "b", 0.8),
        ("e1", "eval", "a", 1),
        ("e2", "eval", "b", 0),
    ):
        rows.append(
            {
                "id": row_id,
                "split": split,
                "segment": "s",
                "action": action,
                "reward": reward,
                "behavior": {"a": 0.5, "b": 0.5},
            }
        )
    return {
        "version": 1,
        "actions": ["a", "b"],
        "rows": rows,
        "policies": {"baseline": {"s": {"a": 0.5, "b": 0.5}}, "candidate": {"s": {"a": 1, "b": 0}}},
    }


class InputTests(unittest.TestCase):
    def test_canonical_order_and_fresh_copies(self):
        raw = small()
        dataset = Dataset.from_document(raw)
        raw["rows"].reverse()
        raw["actions"].reverse()
        self.assertEqual(dataset.sha256, Dataset.from_document(raw).sha256)
        raw["rows"][0]["reward"] = 0.123
        self.assertNotEqual(dataset.sha256, Dataset.from_document(raw).sha256)
        detached = dataset.document
        detached["rows"].clear()
        self.assertEqual(len(dataset.document["rows"]), 4)

    def test_json_round_trip(self):
        dataset = Dataset.from_document(small())
        self.assertEqual(dataset, Dataset.from_json(dataset.encoded))

    def test_duplicate_json_keys(self):
        with self.assertRaisesRegex(ValueError, "duplicate JSON"):
            Dataset.from_json('{"version":1,"version":1}')

    def test_json_size_depth_and_syntax(self):
        for raw in ("x" * 2_000_001, "[" * 2000, "{invalid", "null"):
            with self.subTest(raw=raw[:20]), self.assertRaises(ValueError):
                Dataset.from_json(raw)

    def test_reward_rejects_nonfinite_bool_range_and_huge_integer(self):
        for reward in (True, "1", float("nan"), float("inf"), -0.01, 1.1, 10**1000):
            raw = small()
            raw["rows"][0]["reward"] = reward
            with self.subTest(reward=repr(reward)[:20]), self.assertRaises(ValueError):
                Dataset.from_document(raw)

    def test_distribution_validation(self):
        for value in (
            {"a": 0.2, "b": 0.7},
            {"a": True, "b": 0},
            {"a": 1e-7, "b": 1 - 1e-7},
            {"a": -1, "b": 2},
            {"a": float("nan"), "b": 1},
            {"a": 10**1000, "b": 0},
            {"a": 1},
            {"a": 1, "b": 0, "c": 0},
        ):
            with self.subTest(value=str(value)[:30]), self.assertRaises(ValueError):
                distribution(value, ["a", "b"])

    def test_duplicate_ids_across_splits(self):
        raw = small()
        raw["rows"][0]["id"] = raw["rows"][2]["id"]
        with self.assertRaisesRegex(ValueError, "globally unique"):
            Dataset.from_document(raw)

    def test_zero_propensity_for_observed_action(self):
        raw = small()
        raw["rows"][0]["behavior"] = {"a": 0, "b": 1}
        with self.assertRaisesRegex(ValueError, "zero logging"):
            Dataset.from_document(raw)

    def test_exact_schema_unknown_segment_action_split(self):
        mutations = [
            lambda d: d.update(extra=1),
            lambda d: d["rows"][0].update(extra=1),
            lambda d: d["rows"][0].update(action="c"),
            lambda d: d["rows"][0].update(split="test"),
            lambda d: d["rows"][0].update(segment="unknown"),
            lambda d: d["rows"][0].update(id="bad/name"),
            lambda d: d.update(version=True),
            lambda d: d["actions"].append("a"),
        ]
        for mutation in mutations:
            raw = small()
            mutation(raw)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                Dataset.from_document(raw)

    def test_fit_and_eval_are_required(self):
        for split in ("fit", "eval"):
            raw = small()
            for row in raw["rows"]:
                row["split"] = split
            with self.assertRaises(ValueError):
                Dataset.from_document(raw)


class EstimatorTests(unittest.TestCase):
    def setUp(self):
        self.dataset = Dataset.from_document(small())

    def test_fit_cell_means(self):
        model = fit(self.dataset)
        self.assertEqual(model.predict("s", "a"), 0.2)
        self.assertEqual(model.predict("s", "b"), 0.8)
        self.assertEqual(model.predict("new", "a"), 0.5)
        self.assertFalse(model.observed("new", "a"))

    def test_holdout_rewards_cannot_change_model(self):
        raw = small()
        for row in raw["rows"]:
            if row["split"] == "eval":
                row["reward"] = 0.321
                row["behavior"] = {"a": 0.9, "b": 0.1}
        self.assertEqual(fit(self.dataset), fit(Dataset.from_document(raw)))

    def test_policies_cannot_change_model(self):
        raw = small()
        raw["policies"]["candidate"]["s"] = {"a": 0, "b": 1}
        self.assertEqual(fit(self.dataset), fit(Dataset.from_document(raw)))

    def test_hand_calculated_estimators(self):
        scores, fallback = contributions(self.dataset, fit(self.dataset), "candidate")
        self.assertEqual(fallback, 0)
        self.assertEqual(scores[0][:3], (2, 2, 0.2))
        self.assertAlmostEqual(scores[0][3], 1.8)
        self.assertEqual(scores[1], (0, 0, 0.2, 0.2))
        report = summarize(scores)
        self.assertEqual(report["estimates"], {"dm": 0.2, "ips": 1, "snips": 1, "dr": 1})
        self.assertEqual(report["diagnostics"]["effective_sample_size"], 1)

    def test_logging_policy_ips_equals_observed_mean(self):
        scores, _ = contributions(self.dataset, fit(self.dataset), "baseline")
        report = summarize(scores)
        self.assertEqual(report["estimates"]["ips"], 0.5)
        self.assertEqual(report["estimates"]["snips"], 0.5)
        self.assertEqual(report["diagnostics"]["effective_sample_size"], 2)

    def test_unsupported_action_rejected_even_if_never_logged(self):
        raw = small()
        raw["rows"][3]["action"] = "a"
        for row in raw["rows"]:
            if row["split"] == "eval":
                row["behavior"] = {"a": 1, "b": 0}
        raw["policies"]["candidate"]["s"] = {"a": 0.5, "b": 0.5}
        with self.assertRaisesRegex(ValueError, "unsupported policy"):
            compare(Dataset.from_document(raw), resamples=50)

    def test_zero_weight_snips_is_undefined(self):
        report = summarize([(0, 0, 0.7, 0.7)] * 3)
        self.assertIsNone(report["estimates"]["snips"])
        self.assertEqual(report["diagnostics"]["effective_sample_size"], 0)
        self.assertAlmostEqual(report["estimates"]["dr"], 0.7)

    def test_no_hidden_clipping(self):
        report = summarize([(100, 100, 0.5, 50.5), (0, 0, 0.5, 0.5)])
        self.assertEqual(report["estimates"]["ips"], 50)
        self.assertEqual(report["estimates"]["dr"], 25.5)

    def test_row_specific_propensities_used_individually(self):
        raw = small()
        raw["rows"][2]["behavior"] = {"a": 0.25, "b": 0.75}
        raw["rows"][3]["behavior"] = {"a": 0.9, "b": 0.1}
        dataset = Dataset.from_document(raw)
        scores, _ = contributions(dataset, fit(dataset), "baseline")
        self.assertEqual(scores[0][0], 2)
        self.assertEqual(scores[1][0], 5)
        report = summarize(scores)
        self.assertEqual(report["estimates"]["ips"], 1)
        self.assertAlmostEqual(report["estimates"]["snips"], 2 / 7)
        self.assertAlmostEqual(report["diagnostics"]["effective_sample_size"], 49 / 29)

    def test_exact_action_enumeration_unbiased_ips_dr(self):
        # Enumerate every possible logged action; no Monte Carlo or same-code oracle.
        rng = random.Random(937)
        for _ in range(80):
            p = rng.uniform(0.05, 0.95)
            target_a = rng.random()
            rewards = [rng.random(), rng.random()]
            q = [rng.random(), rng.random()]
            expected = target_a * rewards[0] + (1 - target_a) * rewards[1]
            means = [0.0, 0.0]
            for action, behavior in enumerate((p, 1 - p)):
                raw = small()
                for row in raw["rows"]:
                    if row["split"] == "eval":
                        row.update(
                            action=("a", "b")[action],
                            reward=rewards[action],
                            behavior={"a": p, "b": 1 - p},
                        )
                raw["policies"]["candidate"]["s"] = {"a": target_a, "b": 1 - target_a}
                model = RewardModel((("s", "a", q[0], 1), ("s", "b", q[1], 1)), 0.5, "oracle-test")
                scores, _ = contributions(Dataset.from_document(raw), model, "candidate")
                estimates = summarize(scores)["estimates"]
                means[0] += behavior * estimates["ips"]
                means[1] += behavior * estimates["dr"]
            for value in means:
                self.assertAlmostEqual(value, expected, places=12)

    def test_correct_reward_model_cancels_residual_with_wrong_propensity(self):
        # Deterministic rewards equal Q: correction is zero for either selected arm.
        raw = small()
        for row in raw["rows"]:
            if row["split"] == "eval":
                row["reward"] = 0.2 if row["action"] == "a" else 0.8
                row["behavior"] = {"a": 0.9, "b": 0.1}
        dataset = Dataset.from_document(raw)
        scores, _ = contributions(dataset, fit(dataset), "candidate")
        self.assertEqual(summarize(scores)["estimates"]["dr"], 0.2)

    def test_full_two_row_enumeration_ips_expected_value(self):
        # Both eval draws independently take a/b, giving four complete datasets.
        expected_value = 0
        for arms in itertools.product(("a", "b"), repeat=2):
            raw = small()
            probability = 1
            for row, arm in zip(raw["rows"][2:], arms):
                row.update(
                    action=arm, reward=1 if arm == "a" else 0, behavior={"a": 0.25, "b": 0.75}
                )
                probability *= 0.25 if arm == "a" else 0.75
            dataset = Dataset.from_document(raw)
            scores, _ = contributions(dataset, fit(dataset), "candidate")
            expected_value += probability * summarize(scores)["estimates"]["ips"]
        self.assertAlmostEqual(expected_value, 1)


class ComparisonTests(unittest.TestCase):
    def test_paired_identical_policies_zero_difference(self):
        raw = small()
        raw["policies"]["candidate"] = copy.deepcopy(raw["policies"]["baseline"])
        report = compare(Dataset.from_document(raw), resamples=60)
        for item in report["candidate_minus_baseline"].values():
            self.assertEqual(item["estimate"], 0)
            self.assertEqual(item["percentile_95"], [0, 0])

    def test_undefined_bootstraps_not_silently_dropped(self):
        report = compare(Dataset.from_document(small()), resamples=100)
        item = report["policies"]["candidate"]["intervals"]["snips"]
        self.assertGreater(item["undefined_resamples"], 0)
        self.assertIsNone(item["percentile_95"])
        self.assertIsNone(report["candidate_minus_baseline"]["snips"]["percentile_95"])
        self.assertIn("undefined_bootstrap_estimate", report["policies"]["candidate"]["warnings"])

    def test_reproducible_and_seed_sensitive(self):
        dataset = Dataset.from_document(small())
        a = compare(dataset, resamples=100, seed=1)
        self.assertEqual(a, compare(dataset, resamples=100, seed=1))
        b = compare(dataset, resamples=100, seed=2)
        self.assertEqual(a["reward_model"], b["reward_model"])
        self.assertNotEqual(a["bootstrap"], b["bootstrap"])

    def test_eval_row_reordering_invariant(self):
        raw = small()
        first = compare(Dataset.from_document(raw), resamples=50)
        raw["rows"].reverse()
        self.assertEqual(first, compare(Dataset.from_document(raw), resamples=50))

    def test_fallback_and_missing_target_cell_visible(self):
        raw = small()
        raw["rows"][1]["action"] = "a"
        raw["rows"][3]["action"] = "a"
        raw["policies"]["candidate"]["s"] = {"a": 0, "b": 1}
        report = compare(Dataset.from_document(raw), resamples=50)["policies"]["candidate"]
        self.assertIn("reward_model_fallback_used", report["warnings"])
        self.assertIn("target_cell_not_observed_in_eval", report["warnings"])
        self.assertEqual(report["diagnostics"]["mean_target_mass_using_fallback"], 1)
        self.assertEqual(
            report["diagnostics"]["target_cells_without_eval_observations"],
            [{"segment": "s", "action": "b"}],
        )

    def test_possible_weight_flag_even_without_observed_rare_action(self):
        raw = small()
        for row in raw["rows"]:
            if row["split"] == "eval":
                row.update(action="a", behavior={"a": 0.999, "b": 0.001})
        raw["policies"]["candidate"]["s"] = {"a": 0.5, "b": 0.5}
        report = compare(Dataset.from_document(raw), resamples=50)["policies"]["candidate"]
        self.assertLess(report["diagnostics"]["max_weight"], 1)
        self.assertEqual(report["diagnostics"]["max_possible_weight_on_eval_contexts"], 500)
        self.assertIn("large_importance_weight", report["warnings"])

    def test_validation_parameters_and_unknown_policy(self):
        dataset = Dataset.from_document(small())
        for args in (
            {"resamples": True},
            {"resamples": 49},
            {"resamples": 2001},
            {"seed": -1},
            {"seed": True},
            {"seed": 2**64},
            {"candidate": "absent"},
            {"candidate": "baseline"},
        ):
            with self.subTest(args=args), self.assertRaises(ValueError):
                compare(dataset, **args)

    def test_by_segment_counts(self):
        report = compare(fixture(), resamples=50)
        for policy in report["policies"].values():
            self.assertEqual(
                sum(g["diagnostics"]["rows"] for g in policy["by_segment"].values()), 600
            )

    def test_percentile_interpolation(self):
        self.assertEqual(percentile([0, 10], 0.25), 2.5)
        self.assertEqual(percentile([7], 0.975), 7)


class IntegrationTests(unittest.TestCase):
    def test_synthetic_truth_is_not_in_input_and_contexts_match(self):
        clean, thin = fixture(), fixture("thin")
        self.assertAlmostEqual(truth(clean, "candidate"), 0.8842)
        self.assertEqual(truth(clean, "candidate"), truth(thin, "candidate"))
        for row in clean.document["rows"]:
            self.assertEqual(set(row), {"id", "split", "segment", "action", "reward", "behavior"})

    def test_stress_case_exposes_weight_instability(self):
        report = compare(fixture("thin"), resamples=50)["policies"]["candidate"]
        self.assertLess(report["diagnostics"]["effective_sample_size"], 2)
        self.assertGreater(report["estimates"]["dr"], 1)
        self.assertIn("large_importance_weight", report["warnings"])

    def test_unsupported_synthetic_refused(self):
        with self.assertRaisesRegex(ValueError, "unsupported"):
            compare(fixture("unsupported"), resamples=50)

    def test_cli_valid_compare_and_no_source_change(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.json"
            original = json.dumps(small())
            path.write_text(original, encoding="utf-8")
            out = io.StringIO()
            with redirect_stdout(out):
                code = main(["compare", str(path), "--resamples", "50"])
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(out.getvalue())["baseline"], "baseline")
            self.assertEqual(path.read_text(encoding="utf-8"), original)
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_cli_errors_no_partial_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            for raw in (b"{", b"\xff", b" " * 2_000_001, fixture("unsupported").encoded.encode()):
                path.write_bytes(raw)
                out, err = io.StringIO(), io.StringIO()
                with redirect_stdout(out), redirect_stderr(err):
                    code = main(["compare", str(path), "--resamples", "50"])
                self.assertEqual(code, 2)
                self.assertEqual(out.getvalue(), "")
                self.assertIn("policycompass:", err.getvalue())

    def test_module_fixture_subprocess(self):
        result = subprocess.run(
            [sys.executable, "-m", "policycompass", "fixture"],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(len(Dataset.from_json(result.stdout).document["rows"]), 900)


if __name__ == "__main__":
    unittest.main()
