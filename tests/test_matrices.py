"""Tests for the presence / transition / error matrix layer."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stem.matrices import (  # noqa: E402
    transition_matrix, error_matrix, prescribed_mask, shot_structure,
    presence_from_structure, validate_presence_matrix,
)
from span.pattern_library import PATTERNS  # noqa: E402


class TestTransitionMatrix(unittest.TestCase):
    def test_relay(self):
        S = [[1, 0], [1, 1], [0, 1]]
        np.testing.assert_array_equal(transition_matrix(S), [[0, 1], [-1, 0]])

    def test_is_finite_difference(self):
        rng = np.random.default_rng(0)
        S = rng.integers(0, 2, size=(3, 3))
        np.testing.assert_array_equal(transition_matrix(S), np.diff(S, axis=0))

    def test_shape(self):
        S = np.zeros((5, 4), dtype=int)
        self.assertEqual(transition_matrix(S).shape, (4, 4))

    def test_values_in_range(self):
        """Binary presence can only ever produce -1, 0, +1 transitions."""
        rng = np.random.default_rng(1)
        for _ in range(50):
            S = rng.integers(0, 2, size=(3, 3))
            self.assertTrue(np.isin(transition_matrix(S), [-1, 0, 1]).all())

    def test_rejects_single_shot(self):
        with self.assertRaises(ValueError):
            transition_matrix([[1, 0]])

    def test_rejects_1d(self):
        with self.assertRaises(ValueError):
            transition_matrix([1, 0, 1])


class TestErrorMatrix(unittest.TestCase):
    def test_perfect_is_zero(self):
        dS = np.array([[0, 1], [-1, 0]])
        np.testing.assert_array_equal(error_matrix(dS, dS), np.zeros_like(dS))

    def test_reversal_has_magnitude_two(self):
        """A transition run backwards is the only way to reach |E| = 2."""
        E = error_matrix([[-1, 0]], [[1, 0]])
        self.assertEqual(abs(E[0, 0]), 2)

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            error_matrix([[0, 1]], [[0, 1, 0]])


class TestPrescribedMask(unittest.TestCase):
    def test_counts_match_pattern_library(self):
        expected = {"Relay": 2, "Split": 3, "Accumulation": 2, "Convergence": 3,
                    "Sliding_Window": 3, "Reduction": 2, "Reverse_Relay": 2}
        for name, spec in PATTERNS.items():
            self.assertEqual(
                int(prescribed_mask(spec["delta_S_star"]).sum()), expected[name], name
            )

    def test_zero_cells_excluded(self):
        mask = prescribed_mask([[0, 1], [-1, 0]])
        np.testing.assert_array_equal(mask, [[False, True], [True, False]])


class TestShotStructure(unittest.TestCase):
    def test_relay_notation(self):
        self.assertEqual(shot_structure([[1, 0], [1, 1], [0, 1]]), ["A", "AB", "B"])

    def test_empty_shot(self):
        self.assertEqual(shot_structure([[0, 0], [1, 1]]), ["-", "AB"])

    def test_round_trip_all_patterns(self):
        for name, spec in PATTERNS.items():
            S = spec["S_star"]
            ids = [chr(ord("A") + j) for j in range(S.shape[1])]
            back = presence_from_structure(shot_structure(S, ids), ids)
            np.testing.assert_array_equal(back, S, err_msg=name)

    def test_unknown_entity_raises(self):
        with self.assertRaises(ValueError):
            presence_from_structure(["AZ"], ["A", "B"])


class TestValidation(unittest.TestCase):
    def test_accepts_binary(self):
        out = validate_presence_matrix([[1, 0], [0, 1]])
        self.assertEqual(out.dtype, np.dtype(int))

    def test_rejects_non_binary(self):
        with self.assertRaises(ValueError):
            validate_presence_matrix([[2, 0], [0, 1]])

    def test_rejects_wrong_entity_count(self):
        with self.assertRaises(ValueError):
            validate_presence_matrix([[1, 0], [0, 1]], n_entities=3)

    def test_rejects_wrong_shot_count(self):
        with self.assertRaises(ValueError):
            validate_presence_matrix([[1, 0], [0, 1]], n_shots=3)


class TestPatternLibraryConsistency(unittest.TestCase):
    def test_delta_equals_diff_of_presence(self):
        """ΔS* must be exactly diff(S*) for every canonical pattern."""
        for name, spec in PATTERNS.items():
            np.testing.assert_array_equal(
                transition_matrix(spec["S_star"]), spec["delta_S_star"], err_msg=name
            )

    def test_scenario_counts_sum_to_1000(self):
        self.assertEqual(sum(p["scenarios"] for p in PATTERNS.values()), 1000)


if __name__ == "__main__":
    unittest.main()
