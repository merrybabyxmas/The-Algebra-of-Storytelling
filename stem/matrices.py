#!/usr/bin/env python3
"""
STEM Pipeline: Presence / Transition / Error matrices.

The algebraic core of STEM, kept free of any VLM dependency so that
matrices can be built from *any* presence source (Qwen2-VL, another
detector, or human annotation).

Definitions are those of the paper's evaluation script
(eval_vlm_fullscale.py lines 164-165):

    ΔS_obs = S_obs[1:] - S_obs[:-1]          (finite difference over shots)
    E      = ΔS_obs - ΔS*                    (error matrix)

Shapes, for T shots and N entities:
    S_obs, S*   :  T x N   binary
    ΔS_obs, ΔS* : (T-1) x N   in {-1, 0, +1}
    E           : (T-1) x N   in {-2, ..., +2}
"""
import numpy as np


def transition_matrix(S):
    """
    ΔS = S[1:] - S[:-1].

    From eval_vlm_fullscale.py line 164 (`np.diff(S_obs, axis=0)`).
    """
    S = np.asarray(S, dtype=int)
    if S.ndim != 2:
        raise ValueError(f"presence matrix must be 2-D (T x N), got shape {S.shape}")
    if S.shape[0] < 2:
        raise ValueError(f"need at least 2 shots to form transitions, got {S.shape[0]}")
    return np.diff(S, axis=0)


def error_matrix(delta_S_obs, delta_S_star):
    """
    E = ΔS_obs - ΔS*.

    From eval_vlm_fullscale.py line 165.
    """
    delta_S_obs = np.asarray(delta_S_obs, dtype=int)
    delta_S_star = np.asarray(delta_S_star, dtype=int)
    if delta_S_obs.shape != delta_S_star.shape:
        raise ValueError(
            f"transition matrix shape mismatch: observed {delta_S_obs.shape} "
            f"vs prescribed {delta_S_star.shape}"
        )
    return delta_S_obs - delta_S_star


def prescribed_mask(delta_S_star):
    """
    Boolean mask of the cells PTM is scored on: the non-zero cells of ΔS*.

    From eval_vlm_fullscale.py line 167 (`np.abs(delta_S) > 0`).
    """
    return np.abs(np.asarray(delta_S_star, dtype=int)) > 0


def validate_presence_matrix(S, n_entities=None, n_shots=None, name="S"):
    """Check that a presence matrix is binary and correctly shaped."""
    S = np.asarray(S)
    if S.ndim != 2:
        raise ValueError(f"{name} must be 2-D (T x N), got shape {S.shape}")
    if not np.isin(S, [0, 1]).all():
        bad = sorted(set(np.unique(S).tolist()) - {0, 1})
        raise ValueError(f"{name} must be binary; found values {bad}")
    if n_shots is not None and S.shape[0] != n_shots:
        raise ValueError(f"{name} has {S.shape[0]} shots, expected {n_shots}")
    if n_entities is not None and S.shape[1] != n_entities:
        raise ValueError(f"{name} has {S.shape[1]} entities, expected {n_entities}")
    return S.astype(int)


def shot_structure(S, entity_ids=None):
    """
    Render a presence matrix as the paper's compact notation.

    >>> shot_structure([[1, 0], [1, 1], [0, 1]])
    ['A', 'AB', 'B']
    """
    S = np.asarray(S, dtype=int)
    if entity_ids is None:
        entity_ids = [chr(ord("A") + j) for j in range(S.shape[1])]
    return [
        "".join(entity_ids[j] for j in range(S.shape[1]) if S[t, j] == 1) or "-"
        for t in range(S.shape[0])
    ]


def presence_from_structure(structure, entity_ids):
    """Inverse of `shot_structure`: 'A', 'AB', 'B' -> [[1,0],[1,1],[0,1]]."""
    index = {e: j for j, e in enumerate(entity_ids)}
    S = np.zeros((len(structure), len(entity_ids)), dtype=int)
    for t, shot in enumerate(structure):
        if shot == "-":
            continue
        for token in shot:
            if token not in index:
                raise ValueError(f"unknown entity {token!r} in shot {t+1} ({shot!r})")
            S[t, index[token]] = 1
    return S


if __name__ == "__main__":
    S_star = [[1, 0], [1, 1], [0, 1]]          # Relay: A -> AB -> B
    S_obs = [[1, 0], [1, 1], [1, 1]]           # A fails to exit

    dS_star = transition_matrix(S_star)
    dS_obs = transition_matrix(S_obs)
    E = error_matrix(dS_obs, dS_star)

    print("=== STEM Matrices (Relay example) ===")
    print(f"structure S*  : {shot_structure(S_star)}")
    print(f"structure S_obs: {shot_structure(S_obs)}")
    print(f"ΔS*   =\n{dS_star}")
    print(f"ΔS_obs=\n{dS_obs}")
    print(f"E     =\n{E}")
    print(f"prescribed cells: {prescribed_mask(dS_star).sum()}")
