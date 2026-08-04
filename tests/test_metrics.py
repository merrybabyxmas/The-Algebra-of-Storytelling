"""Tests for PTM / EPA / Frobenius."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stem.metrics import compute_metrics, ptm, epa, cohens_d  # noqa: E402
from span.pattern_library import PATTERNS  # noqa: E402

RELAY = PATTERNS["Relay"]
S_STAR = RELAY["S_star"]
DELTA_STAR = RELAY["delta_S_star"]


class TestPTM(unittest.TestCase):
    def test_perfect(self):
        r = compute_metrics(S_STAR, S_STAR, DELTA_STAR)
        self.assertEqual(r["match_prescribed"], 1.0)
        self.assertEqual(r["epa"], 1.0)
        self.assertEqual(r["frob_dist"], 0.0)

    def test_half(self):
        """B enters as prescribed, A never exits: 1 of 2 prescribed cells."""
        r = compute_metrics([[1, 0], [1, 1], [1, 1]], S_STAR, DELTA_STAR)
        self.assertAlmostEqual(r["match_prescribed"], 0.5)

    def test_zero(self):
        r = compute_metrics([[0, 0], [0, 0], [0, 0]], S_STAR, DELTA_STAR)
        self.assertEqual(r["match_prescribed"], 0.0)

    def test_denominator_is_prescribed_cell_count(self):
        """PTM only ever takes values k / n_prescribed."""
        for name, spec in PATTERNS.items():
            n = int(np.count_nonzero(spec["delta_S_star"]))
            allowed = {round(k / n, 6) for k in range(n + 1)}
            rng = np.random.default_rng(7)
            S_shape = spec["S_star"].shape
            for _ in range(40):
                S_obs = rng.integers(0, 2, size=S_shape)
                value = round(ptm(S_obs, spec["delta_S_star"]), 6)
                self.assertIn(value, allowed, f"{name}: {value} not a multiple of 1/{n}")

    def test_ignores_unprescribed_cells(self):
        """
        The sparsity property: PTM is blind to errors outside ΔS* != 0.

        In Accumulation, A is prescribed to be present in all three shots,
        so its whole ΔS* column is zero. Dropping A out of the middle shot
        is a visible failure -- EPA falls and it is flagged Type III -- but
        PTM must not move, because no prescribed transition was touched.
        """
        acc = PATTERNS["Accumulation"]
        dS_star = acc["delta_S_star"]
        self.assertTrue((dS_star[:, 0] == 0).all())

        base = np.array(acc["S_star"])
        perturbed = base.copy()
        perturbed[1, 0] = 0

        self.assertEqual(ptm(base, dS_star), 1.0)
        self.assertEqual(ptm(perturbed, dS_star), 1.0)
        self.assertLess(epa(perturbed, base), 1.0)


class TestEPA(unittest.TestCase):
    def test_is_cellwise_accuracy(self):
        S_obs = [[1, 0], [1, 1], [1, 1]]
        self.assertAlmostEqual(epa(S_obs, S_STAR), 5 / 6)

    def test_independent_of_prescription(self):
        self.assertEqual(epa(S_STAR, S_STAR), 1.0)

    def test_epa_high_while_ptm_zero(self):
        """The failure mode the paper is built to expose."""
        S_obs = [[1, 1], [1, 1], [1, 1]]  # both entities always on screen
        r = compute_metrics(S_obs, S_STAR, DELTA_STAR)
        self.assertEqual(r["match_prescribed"], 0.0)
        self.assertGreater(r["epa"], 0.6)


class TestFrobenius(unittest.TestCase):
    def test_zero_when_perfect(self):
        r = compute_metrics(S_STAR, S_STAR, DELTA_STAR)
        self.assertEqual(r["frob_dist"], 0.0)

    def test_matches_numpy(self):
        S_obs = [[1, 0], [1, 1], [1, 1]]
        r = compute_metrics(S_obs, S_STAR, DELTA_STAR)
        expected = np.linalg.norm(np.array(r["delta_S_obs"]) - DELTA_STAR, "fro")
        self.assertAlmostEqual(r["frob_dist"], float(expected))

    def test_matrix_distance_normalised(self):
        S_obs = [[0, 1], [0, 1], [1, 0]]
        r = compute_metrics(S_obs, S_STAR, DELTA_STAR)
        self.assertGreaterEqual(r["matrix_distance"], 0.0)
        self.assertLessEqual(r["matrix_distance"], 1.0)


class TestCohensD(unittest.TestCase):
    def test_zero_for_identical(self):
        a = np.array([0.1, 0.2, 0.3, 0.4])
        self.assertAlmostEqual(cohens_d(a, a), 0.0)

    def test_sign(self):
        a = np.array([1.0, 1.1, 0.9, 1.0])
        b = np.array([0.0, 0.1, -0.1, 0.0])
        self.assertGreater(cohens_d(a, b), 0)
        self.assertLess(cohens_d(b, a), 0)

    def test_no_division_by_zero(self):
        a = np.array([1.0, 1.0, 1.0])
        self.assertEqual(cohens_d(a, a), 0.0)


if __name__ == "__main__":
    unittest.main()
