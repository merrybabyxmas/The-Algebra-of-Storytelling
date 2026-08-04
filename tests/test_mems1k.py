"""Tests for the released MEMS-1K benchmark file and the evaluator glue."""
import json
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stem.evaluator import (  # noqa: E402
    load_benchmark, evaluate_presence, entity_names, prescribed_matrices,
    scenario_id_from_path, DEFAULT_BENCHMARK,
)
from stem.matrices import shot_structure  # noqa: E402
from span.pattern_library import PATTERNS  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLITS = os.path.join(REPO, "mems1k", "splits")


@unittest.skipUnless(os.path.exists(DEFAULT_BENCHMARK), "benchmark file missing")
class TestBenchmarkFile(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bench = load_benchmark()

    def test_size(self):
        self.assertEqual(len(self.bench), 1000)

    def test_pattern_counts(self):
        counts = {}
        for s in self.bench.values():
            counts[s["pattern"]] = counts.get(s["pattern"], 0) + 1
        for name, spec in PATTERNS.items():
            self.assertEqual(counts[name], spec["scenarios"], name)

    def test_matrices_are_canonical(self):
        for s in self.bench.values():
            S, dS = prescribed_matrices(s)  # raises if it drifted
            np.testing.assert_array_equal(np.diff(S, axis=0), dS, s["scenario_id"])

    def test_entity_column_alignment(self):
        for s in self.bench.values():
            ids = [e["entity_id"] for e in s["entities"]]
            self.assertEqual(ids, sorted(ids), s["scenario_id"])
            self.assertEqual(len(ids), len(s["presence_matrix"][0]), s["scenario_id"])

    def test_shot_structure_matches_presence(self):
        for s in self.bench.values():
            ids = [e["entity_id"] for e in s["entities"]]
            self.assertEqual(
                shot_structure(s["presence_matrix"], ids), s["shot_structure"],
                s["scenario_id"],
            )

    def test_prompts_present(self):
        for s in self.bench.values():
            for k in ("shot_1", "shot_2", "shot_3"):
                self.assertTrue(s["gen_prompt"][k].strip(), s["scenario_id"])
                self.assertTrue(s["eval_prompt"][k].strip(), s["scenario_id"])

    def test_eval_prompt_is_gen_prompt_without_suffix(self):
        for s in self.bench.values():
            suffix = s["generation_settings"]["positive_suffix"] or ""
            for k in ("shot_1", "shot_2", "shot_3"):
                gen, ev = s["gen_prompt"][k], s["eval_prompt"][k]
                self.assertIn(gen, (ev + suffix, ev), f"{s['scenario_id']}/{k}")

    def test_entity_names_nonempty(self):
        for s in self.bench.values():
            for e in s["entities"]:
                self.assertTrue(e["name"].strip(), s["scenario_id"])
                self.assertTrue(e["eval_name"].strip(), s["scenario_id"])

    def test_active_transition_count(self):
        for s in self.bench.values():
            self.assertEqual(
                int(np.count_nonzero(s["transition_matrix"])),
                s["metadata"]["num_active_transitions"],
                s["scenario_id"],
            )


@unittest.skipUnless(os.path.isdir(SPLITS), "splits missing")
class TestSplits(unittest.TestCase):
    def test_splits_partition_the_benchmark(self):
        bench = load_benchmark()
        seen = set()
        for name, spec in PATTERNS.items():
            path = os.path.join(SPLITS, f"{name.lower()}.json")
            with open(path) as f:
                payload = json.load(f)
            self.assertEqual(len(payload["scenarios"]), spec["scenarios"], name)
            for s in payload["scenarios"]:
                self.assertEqual(s["pattern"], name)
                self.assertIn(s["scenario_id"], bench)
                seen.add(s["scenario_id"])
        self.assertEqual(len(seen), len(bench))


@unittest.skipUnless(os.path.exists(DEFAULT_BENCHMARK), "benchmark file missing")
class TestEvaluatePresence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bench = load_benchmark()
        cls.scenario = cls.bench["DM1K_777_Relay_001"]

    def test_oracle_scores_perfect(self):
        for s in list(self.bench.values())[:50]:
            rec = evaluate_presence(s, s["presence_matrix"])
            self.assertEqual(rec["ptm"], 1.0, s["scenario_id"])
            self.assertEqual(rec["epa"], 1.0, s["scenario_id"])
            self.assertEqual(rec["failure_types"], ["None"], s["scenario_id"])

    def test_record_shape(self):
        rec = evaluate_presence(self.scenario, self.scenario["presence_matrix"],
                                method="oracle")
        for key in ("scenario_id", "pattern", "entities", "S_obs", "S_ideal",
                    "delta_S_obs", "delta_S_ideal", "E_error", "ptm", "epa",
                    "frob_dist", "failures", "failure_types", "failure_cells",
                    "method"):
            self.assertIn(key, rec)

    def test_match_prescribed_alias(self):
        rec = evaluate_presence(self.scenario, self.scenario["presence_matrix"])
        self.assertEqual(rec["ptm"], rec["match_prescribed"])

    def test_entity_names_order(self):
        names = entity_names(self.scenario)
        self.assertEqual(names, [e["eval_name"] for e in self.scenario["entities"]])

    def test_rejects_wrong_shape(self):
        with self.assertRaises(ValueError):
            evaluate_presence(self.scenario, [[1, 0, 0], [1, 1, 0], [0, 1, 0]])

    def test_rejects_non_binary(self):
        with self.assertRaises(ValueError):
            evaluate_presence(self.scenario, [[1, 0], [1, 2], [0, 1]])

    def test_rejects_drifted_matrices(self):
        broken = dict(self.scenario)
        broken["presence_matrix"] = [[0, 0], [0, 0], [0, 0]]
        with self.assertRaises(ValueError):
            evaluate_presence(broken, self.scenario["presence_matrix"])


class TestPathHelpers(unittest.TestCase):
    def test_scenario_id_from_path(self):
        self.assertEqual(
            scenario_id_from_path("/videos/Relay/DM1K_777_Relay_001.mp4"),
            "DM1K_777_Relay_001",
        )


if __name__ == "__main__":
    unittest.main()
