"""
SPAN Framework: Linear Operators for Global Narrative Composition

Implements the four operators from Appendix B:
  - Column permutation σ ∈ Σ_N
  - Time reversal T
  - Spatial expansion Π_N
  - Temporal shift Φ_τ
"""
import numpy as np
from itertools import permutations


# ============================================================
# Column Permutation σ ∈ Σ_N
# ============================================================

def column_permute(M, perm):
    """
    Apply column permutation σ to matrix M.
    σ reassigns entity roles by reordering columns.

    Args:
        M: 2D numpy array
        perm: tuple/list of column indices, e.g., (2, 0, 1)
    Returns:
        Permuted matrix
    """
    return M[:, list(perm)]


def all_column_permutations(N):
    """Return all N! permutations of N entity indices."""
    return list(permutations(range(N)))


# ============================================================
# Time Reversal Operator T
# ============================================================

def time_reversal(M):
    """
    Time reversal operator T: reverse temporal order and negate signs.

    Rewinding a narrative turns entrances into exits and exits into entrances,
    so T(ΔS)_t,e = -ΔS_{T-t,e}.
    """
    M = np.asarray(M)
    return -M[::-1].copy()


# ============================================================
# Spatial Expansion Operator Π_N (Appendix B.1)
# ============================================================

def spatial_expand(X, N):
    """
    Π_N: Append (N-3) zero columns to a 2×3 local matrix.
    Embeds the canonical local space into the global entity space.

    Property: Π_N(c1*X + c2*Y) = c1*Π_N(X) + c2*Π_N(Y)  (linear)

    Args:
        X: 2×3 numpy array (local basis matrix)
        N: target number of entities (≥ 3)
    Returns:
        2×N numpy array
    """
    assert X.shape == (2, 3), f"Expected 2×3, got {X.shape}"
    assert N >= 3, f"N must be ≥ 3, got {N}"
    if N == 3:
        return X.copy()
    padding = np.zeros((2, N - 3), dtype=X.dtype)
    return np.hstack([X, padding])


# ============================================================
# Temporal Shift Operator Φ_τ (Appendix B.1)
# ============================================================

def temporal_shift(X, tau, T_shots):
    """
    Φ_τ: Insert a 2×N block at temporal position τ in a (T-1)×N
    global transition matrix, padding with zeros above and below.

    Args:
        X: 2×N numpy array (local transition block)
        tau: temporal window index (1-indexed, 1 ≤ τ ≤ T-2)
        T_shots: total number of shots
    Returns:
        (T-1)×N numpy array
    """
    N = X.shape[1]
    T_minus_1 = T_shots - 1
    assert 1 <= tau <= T_shots - 2, f"τ={tau} out of range [1, {T_shots-2}]"

    result = np.zeros((T_minus_1, N), dtype=X.dtype)
    result[tau - 1:tau + 1, :] = X
    return result


# ============================================================
# Grand Spanning Composition (Theorem B.1)
# ============================================================

def compose_global(basis_terms, T_shots, N):
    """
    Reconstruct a global ΔS from localized basis operations.

    ΔS = Σ_τ Σ_m c_{τ,m} · (Φ_τ ∘ σ_{τ,m} ∘ Π_N)(B_m)

    Args:
        basis_terms: list of (c, tau, sigma, B_m) tuples
            c: integer coefficient
            tau: temporal window (1-indexed)
            sigma: column permutation (tuple of N indices)
            B_m: 2×3 basis matrix
        T_shots: total number of shots
        N: total number of entities
    Returns:
        (T-1)×N global transition matrix
    """
    T_minus_1 = T_shots - 1
    result = np.zeros((T_minus_1, N), dtype=int)

    for c, tau, sigma, B_m in basis_terms:
        expanded = spatial_expand(B_m, N)
        permuted = column_permute(expanded, sigma)
        shifted = temporal_shift(permuted, tau, T_shots)
        result += c * shifted

    return result


if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from span.basis_matrices import B1, B2, J

    print("=== Operator Demonstrations ===\n")

    # Spatial expansion example (Appendix B, Example 1.1)
    print("Π_5(B1) =")
    print(spatial_expand(B1, 5))

    # Temporal shift example (Appendix B, Example 2.1)
    B1_hat = spatial_expand(B1, 5)
    print(f"\nΦ_2(B1_hat) for T=5 shots:")
    print(temporal_shift(B1_hat, 2, 5))

    # Worked example: 4-shot relay chain A→AB→BC→C (Appendix B.4)
    print("\n=== Worked Example: A→AB→BC→C ===")

    # Window τ=1: ΔS^(1) = B1 + J_23
    # Window τ=2: ΔS^(2) = -J_22
    terms = [
        (1, 1, (0, 1, 2), B1),        # c=1, τ=1, identity perm, B1
        (1, 1, (0, 1, 2), J(1, 2)),   # c=1, τ=1, identity perm, J_23 (entry +1)
        (-1, 2, (0, 1, 2), J(1, 1)),  # c=-1, τ=2, identity perm, J_22 (exit -1)
    ]

    result = compose_global(terms, T_shots=4, N=3)
    expected = np.array([[0, 1, 0], [-1, 0, 1], [0, -1, 0]])
    print(f"Reconstructed ΔS:\n{result}")
    print(f"Expected ΔS:\n{expected}")
    print(f"Match: {np.array_equal(result, expected)} ✓")
