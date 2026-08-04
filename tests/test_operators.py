"""Tests for SPAN algebraic operators."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from span.basis_matrices import B1, B2  # noqa: E402
from span.operators import (  # noqa: E402
    column_permute, time_reversal, spatial_expand, temporal_shift,
)
from span.pattern_library import PATTERNS  # noqa: E402


class TestColumnPermute(unittest.TestCase):
    def test_reassigns_entity_columns(self):
        M = np.array([[1, 2, 3], [4, 5, 6]])
        np.testing.assert_array_equal(column_permute(M, (2, 0, 1)),
                                      [[3, 1, 2], [6, 4, 5]])


class TestTimeReversal(unittest.TestCase):
    def test_matches_paper_definition(self):
        M = np.array([[0, -1, 0], [-1, 1, 0]])
        np.testing.assert_array_equal(time_reversal(M), -M[::-1])

    def test_split_reverses_to_role_swapped_convergence(self):
        convergence = np.pad(
            PATTERNS["Convergence"]["delta_S_star"],
            ((0, 0), (0, 1)),
            mode="constant",
        )
        # T(Split: AB -> A -> B) is B -> A -> AB. The pattern library stores
        # canonical Convergence as A -> B -> AB, so one role swap aligns them.
        np.testing.assert_array_equal(
            time_reversal(B2),
            column_permute(convergence, (1, 0, 2)),
        )

    def test_reverse_relay_is_time_reversed_relay(self):
        reverse_relay = np.pad(
            PATTERNS["Reverse_Relay"]["delta_S_star"],
            ((0, 0), (0, 1)),
            mode="constant",
        )
        np.testing.assert_array_equal(time_reversal(B1), reverse_relay)

    def test_time_reversal_is_involution(self):
        M = np.array([[0, 1, -1], [-1, 0, 1]])
        np.testing.assert_array_equal(time_reversal(time_reversal(M)), M)


class TestGlobalOperators(unittest.TestCase):
    def test_spatial_expand_pads_zero_columns(self):
        np.testing.assert_array_equal(
            spatial_expand(B1, 5),
            [[0, 1, 0, 0, 0], [-1, 0, 0, 0, 0]],
        )

    def test_temporal_shift_inserts_two_row_block(self):
        shifted = temporal_shift(spatial_expand(B1, 3), tau=2, T_shots=5)
        expected = np.array([[0, 0, 0], [0, 1, 0], [-1, 0, 0], [0, 0, 0]])
        np.testing.assert_array_equal(shifted, expected)


if __name__ == "__main__":
    unittest.main()
