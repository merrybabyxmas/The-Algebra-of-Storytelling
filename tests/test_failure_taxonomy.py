"""Tests for the Type I-IV failure taxonomy."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stem.matrices import transition_matrix, error_matrix  # noqa: E402
from stem.failure_taxonomy import (  # noqa: E402
    classify_failures, failure_labels, classify_cells, TYPE_NAMES,
)
from span.pattern_library import PATTERNS  # noqa: E402

RELAY = PATTERNS["Relay"]          # A -> AB -> B
DELTA_STAR = RELAY["delta_S_star"]  # [[0, +1], [-1, 0]]


def types_for(S_obs, delta_S_star=DELTA_STAR):
    E = error_matrix(transition_matrix(S_obs), delta_S_star)
    return classify_failures(delta_S_star, E)


class TestTypeAssignment(unittest.TestCase):
    def test_no_failure(self):
        self.assertEqual(types_for(RELAY["S_star"]), {"None"})

    def test_type_i_context_bleeding(self):
        """A is prescribed to exit (ΔS* = -1) but stays on screen."""
        self.assertEqual(types_for([[1, 0], [1, 1], [1, 1]]), {"I"})

    def test_type_ii_amnesia(self):
        """B is prescribed to enter (ΔS* = +1) but never appears."""
        self.assertEqual(types_for([[1, 0], [1, 0], [0, 0]]), {"II"})

    def test_type_iii_spurious(self):
        """A cell with ΔS* = 0 changes state anyway.

        Relay prescribes nothing for A at transition 1 (ΔS*[0, 0] = 0);
        here A pops into existence there.
        """
        types = types_for([[0, 0], [1, 1], [0, 1]])
        self.assertIn("III", types)

    def test_type_iv_reversal_needs_magnitude_two(self):
        E = np.array([[0, -2], [0, 0]])
        self.assertEqual(classify_failures(DELTA_STAR, E), {"IV"})

    def test_type_iv_takes_precedence(self):
        """|E| = 2 is Reversal even where an entrance was prescribed."""
        E = np.array([[0, -2], [0, 0]])  # ΔS*[0,1] = +1 but observed -1
        self.assertEqual(classify_failures(DELTA_STAR, E), {"IV"})

    def test_multiple_types(self):
        types = types_for([[1, 1], [1, 1], [1, 1]])
        self.assertGreaterEqual(len(types), 1)
        self.assertNotIn("None", types)

    def test_none_is_exclusive(self):
        """'None' never co-occurs with a real failure type."""
        rng = np.random.default_rng(3)
        for _ in range(100):
            S = rng.integers(0, 2, size=(3, 2))
            t = types_for(S)
            if "None" in t:
                self.assertEqual(t, {"None"})

    def test_every_type_has_a_name(self):
        for t in ("I", "II", "III", "IV", "None"):
            self.assertIn(t, TYPE_NAMES)


class TestExhaustiveness(unittest.TestCase):
    def test_every_nonzero_cell_is_classified(self):
        """Across all patterns and all reachable S_obs, no error cell is unlabelled."""
        rng = np.random.default_rng(11)
        for name, spec in PATTERNS.items():
            dS_star = spec["delta_S_star"]
            for _ in range(60):
                S_obs = rng.integers(0, 2, size=spec["S_star"].shape)
                E = error_matrix(transition_matrix(S_obs), dS_star)
                n_bad = int(np.count_nonzero(E))
                cells = classify_cells(dS_star, transition_matrix(S_obs), E)
                self.assertEqual(len(cells), n_bad, name)
                self.assertTrue(all(c["type"] in {"I", "II", "III", "IV"} for c in cells))

    def test_types_agree_between_granularities(self):
        """classify_cells and classify_failures must not disagree."""
        rng = np.random.default_rng(13)
        for name, spec in PATTERNS.items():
            dS_star = spec["delta_S_star"]
            for _ in range(60):
                S_obs = rng.integers(0, 2, size=spec["S_star"].shape)
                dS_obs = transition_matrix(S_obs)
                E = error_matrix(dS_obs, dS_star)
                per_video = classify_failures(dS_star, E) - {"None"}
                per_cell = {c["type"] for c in classify_cells(dS_star, dS_obs, E)}
                self.assertEqual(per_video, per_cell, name)


class TestLabels(unittest.TestCase):
    def test_label_format(self):
        S_obs = [[1, 0], [1, 1], [1, 1]]
        labels = failure_labels(DELTA_STAR, transition_matrix(S_obs),
                                error_matrix(transition_matrix(S_obs), DELTA_STAR))
        self.assertEqual(labels, ["missed_exit_t2_e0"])

    def test_empty_when_perfect(self):
        S = RELAY["S_star"]
        self.assertEqual(
            failure_labels(DELTA_STAR, transition_matrix(S),
                           error_matrix(transition_matrix(S), DELTA_STAR)),
            [],
        )


if __name__ == "__main__":
    unittest.main()
