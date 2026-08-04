#!/usr/bin/env python3
"""
STEM Pipeline: Evaluation Metrics (PTM, EPA, Frobenius).

Extracted from the actual evaluation script (eval_vlm_fullscale.py
lines 164-177) and the table generation script (generate_tables.py).
All computations below are identical to those that produced the paper's
results; `scripts/aggregate_results.py` reports them for any set of
records through this module.

    PTM  Prescribed Transition Match
         fraction of *prescribed* cells (ΔS* != 0) that were executed exactly.
         Sparsity-aware: cells the story never asked for do not dilute it.

    EPA  Entity Presence Accuracy
         fraction of all T x N presence cells that match S*.
         Dense: measures the detector-visible state, not the narrative.

    ||E||_F  Frobenius norm of the error matrix; magnitude of total deviation.

Run this file for a demo with synthetic examples.
"""
import numpy as np

from .matrices import transition_matrix, error_matrix, prescribed_mask
from .failure_taxonomy import classify_failures, failure_labels

__all__ = [
    "compute_metrics", "ptm", "epa", "cohens_d",
    "classify_failures", "failure_labels",
]


# ============================================================
# Core Metrics (from eval_vlm_fullscale.py lines 164-177)
# ============================================================

def compute_metrics(S_obs, S_ideal, delta_S_star):
    """
    Compute PTM, EPA, and Frobenius distance for one video.
    Directly extracted from eval_vlm_fullscale.py lines 164-177.

    Args:
        S_obs:        observed presence matrix (T x N), list or np.ndarray
        S_ideal:      prescribed presence matrix S* (T x N)
        delta_S_star: prescribed transition matrix ΔS* ((T-1) x N)
    Returns:
        dict with match_prescribed (PTM), epa, frob_dist, matrix_distance,
        delta_S_obs and E_error -- the same field names as the paper's records.
    """
    S_obs = np.asarray(S_obs)
    S_ideal = np.asarray(S_ideal)
    delta_S_star = np.asarray(delta_S_star)

    # ΔS_obs = S_obs[1:] - S_obs[:-1]  (line 164)
    delta_S_obs = transition_matrix(S_obs)

    # E = ΔS_obs - ΔS*  (line 165)
    E_error = error_matrix(delta_S_obs, delta_S_star)

    # PTM: prescribed transition match (lines 167-169)
    prescribed = prescribed_mask(delta_S_star)
    n_prescribed = prescribed.sum()
    n_correct = np.sum((delta_S_obs == delta_S_star) & prescribed)
    match_prescribed = float(n_correct / max(n_prescribed, 1))

    # Full match over all cells, prescribed or not (line 171)
    full_match = float(np.mean(delta_S_obs == delta_S_star))

    # EPA: entity presence accuracy (line 172)
    epa_val = float(np.mean(S_obs == S_ideal))

    # Frobenius distance (line 173)
    frob_dist = float(np.linalg.norm(delta_S_obs - delta_S_star, "fro"))

    # Normalized matrix distance: 0 = perfect, 1 = worst (lines 176-177)
    max_frob = np.sqrt(2 * delta_S_star.size)
    matrix_distance = float(frob_dist / max_frob) if max_frob > 0 else 0.0

    return {
        "match_prescribed": match_prescribed,  # PTM
        "full_match": full_match,
        "epa": epa_val,
        "frob_dist": frob_dist,
        "matrix_distance": matrix_distance,
        "delta_S_obs": delta_S_obs.tolist(),
        "E_error": E_error.tolist(),
    }


def ptm(S_obs, delta_S_star):
    """PTM alone, for callers that only need the headline metric."""
    delta_S_obs = transition_matrix(S_obs)
    delta_S_star = np.asarray(delta_S_star)
    prescribed = prescribed_mask(delta_S_star)
    n_correct = np.sum((delta_S_obs == delta_S_star) & prescribed)
    return float(n_correct / max(prescribed.sum(), 1))


def epa(S_obs, S_ideal):
    """EPA alone."""
    return float(np.mean(np.asarray(S_obs) == np.asarray(S_ideal)))


# ============================================================
# Statistical Tests (from generate_tables.py lines 56-61)
# ============================================================

def cohens_d(a, b):
    """Cohen's d effect size. From generate_tables.py lines 56-61."""
    a, b = np.asarray(a), np.asarray(b)
    na, nb = len(a), len(b)
    pooled_std = np.sqrt(
        ((na - 1) * a.std(ddof=1) ** 2 + (nb - 1) * b.std(ddof=1) ** 2) / (na + nb - 2)
    )
    if pooled_std == 0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled_std)


if __name__ == "__main__":
    print("=" * 66)
    print("STEM Metrics Demo -- Relay (A -> AB -> B)")
    print("=" * 66)

    S_ideal = [[1, 0], [1, 1], [0, 1]]
    delta_S_star = np.array([[0, +1], [-1, 0]])

    cases = [
        ("Perfect", [[1, 0], [1, 1], [0, 1]]),
        ("Type I",  [[1, 0], [1, 1], [1, 1]]),   # A persists
        ("Type II", [[1, 0], [1, 0], [0, 0]]),   # B never enters
        ("Empty",   [[0, 0], [0, 0], [0, 0]]),   # nothing generated
    ]
    for name, S_obs in cases:
        r = compute_metrics(S_obs, S_ideal, delta_S_star)
        ftypes = sorted(classify_failures(delta_S_star, r["E_error"]))
        print(f"  {name:8s}: PTM={r['match_prescribed']:.3f}  EPA={r['epa']:.3f}  "
              f"||E||_F={r['frob_dist']:.3f}  failures={ftypes}")

    print("\nNote how EPA stays high while PTM collapses: EPA is dense over all")
    print("presence cells, PTM scores only the transitions the story prescribed.")
