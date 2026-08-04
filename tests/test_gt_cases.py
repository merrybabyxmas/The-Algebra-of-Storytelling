"""
Golden regression against the ground-truth cat videos.

`demo/assets/gt_cases.json` prescribes a Relay narrative; the five videos in
`demo/assets/gt_videos/` are cuts of real footage that either follow it or
break it in one specific way. Because the footage is real and the edit is
known, the correct answer is known independently of any model -- so these
cases pin the metric and the taxonomy to reality, not to our own output.

The stored detections in `demo/assets/gt_detected.json` came from
Qwen2-VL-7B and reproduce the paper's GT validation run exactly. These
tests replay them through the pipeline without needing a GPU.
"""
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stem.evaluator import load_benchmark, evaluate_presence  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CASES = os.path.join(REPO, "demo", "assets", "gt_cases.json")
DETECTED = os.path.join(REPO, "demo", "assets", "gt_detected.json")
VIDEOS = os.path.join(REPO, "demo", "assets", "gt_videos")

#: The values the paper reports for these five cases.
GOLDEN = {
    "GT_Perfect":             {"ptm": 1.0, "epa": 1.0,     "types": ["None"]},
    "Type_I_ContextBleeding": {"ptm": 0.5, "epa": 5 / 6,   "types": ["I"]},
    "Type_II_Amnesia":        {"ptm": 0.0, "epa": 0.5,     "types": ["I", "II"]},
    "Type_III_Spurious":      {"ptm": 0.5, "epa": 2 / 3,   "types": ["II", "III"]},
    "Type_IV_Reversal":       {"ptm": 0.5, "epa": 2 / 3,   "types": ["III", "IV"]},
}


@unittest.skipUnless(os.path.exists(CASES) and os.path.exists(DETECTED),
                     "ground-truth demo assets missing")
class TestGroundTruthCases(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bench = load_benchmark(CASES)
        with open(DETECTED) as f:
            cls.detected = json.load(f)["cases"]

    def test_all_five_cases_present(self):
        self.assertEqual(set(self.bench), set(GOLDEN))
        self.assertEqual(set(self.detected), set(GOLDEN))

    def test_metrics_match_the_paper(self):
        for sid, want in GOLDEN.items():
            rec = evaluate_presence(self.bench[sid], self.detected[sid]["S_obs"])
            self.assertAlmostEqual(rec["ptm"], want["ptm"], places=12, msg=sid)
            self.assertAlmostEqual(rec["epa"], want["epa"], places=12, msg=sid)
            self.assertEqual(rec["failure_types"], want["types"], sid)

    def test_each_case_exhibits_its_named_failure(self):
        """Type_I_* must actually produce a Type I, and so on."""
        for sid in GOLDEN:
            if sid == "GT_Perfect":
                continue
            expected_type = sid.split("_")[1]  # 'I', 'II', 'III', 'IV'
            rec = evaluate_presence(self.bench[sid], self.detected[sid]["S_obs"])
            self.assertIn(expected_type, rec["failure_types"], sid)

    def test_scenario_declared_expectations_agree(self):
        """The `expected` block in gt_cases.json must not drift from reality."""
        for sid, scenario in self.bench.items():
            rec = evaluate_presence(scenario, self.detected[sid]["S_obs"])
            self.assertAlmostEqual(rec["ptm"], scenario["expected"]["ptm"], places=12, msg=sid)
            self.assertEqual(rec["failure_types"], scenario["expected"]["failure_types"], sid)

    def test_perfect_case_is_clean(self):
        rec = evaluate_presence(self.bench["GT_Perfect"], self.detected["GT_Perfect"]["S_obs"])
        self.assertEqual(rec["failures"], [])
        self.assertEqual(rec["frob_dist"], 0.0)

    def test_videos_are_shipped(self):
        for sid in GOLDEN:
            self.assertTrue(os.path.exists(os.path.join(VIDEOS, f"{sid}.mp4")), sid)


if __name__ == "__main__":
    unittest.main()
