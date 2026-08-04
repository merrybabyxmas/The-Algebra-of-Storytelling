#!/usr/bin/env python3
"""
STEM Pipeline: 4-type failure taxonomy.

Every non-zero cell of the error matrix E = ΔS_obs - ΔS* is attributed to
exactly one failure type, determined by what the cell *should* have done
(the value of ΔS* there) and how badly it missed:

    Type I   Context Bleeding   ΔS* = -1, entity fails to exit
    Type II  Amnesia            ΔS* = +1, entity fails to enter
    Type III Spurious           ΔS* =  0, entity changes state anyway
    Type IV  Reversal           |E| = 2, the transition ran backwards

Type IV takes precedence: a |E|=2 cell is a complete inversion regardless
of what was prescribed.

Two granularities, both taken from the code that produced the paper:
  * `classify_cells` / `failure_labels` -- per-cell, eval_vlm_fullscale.py
    lines 180-191 (the `failures` field of every result record).
  * `classify_failures` -- per-video set of types, generate_tables.py
    lines 126-148 (Table 4, the failure-distribution table).
"""
import numpy as np

TYPE_NAMES = {
    "I": "Context Bleeding",
    "II": "Amnesia",
    "III": "Spurious Transition",
    "IV": "Reversal",
    "None": "No Failure",
}

#: `failures` string prefix -> taxonomy type, for the per-cell labels
#: emitted by eval_vlm_fullscale.py.
LABEL_TO_TYPE = {
    "missed_exit": "I",
    "missed_entry": "II",
    "spurious": "III",
}


# ============================================================
# Per-video classification (from generate_tables.py lines 126-148)
# ============================================================

def classify_failures(delta_S_star, E_error):
    """
    Return the set of failure types present in one video.

    Extracted verbatim from generate_tables.py `classify_video_failures()`,
    which produced the paper's Table 4.
    """
    delta_S_star = np.asarray(delta_S_star)
    E_error = np.asarray(E_error)
    types = set()
    rows, cols = delta_S_star.shape

    for i in range(rows):
        for j in range(cols):
            ds = delta_S_star[i, j]
            ee = E_error[i, j]
            if ee == 0:
                continue
            if abs(ee) == 2:
                types.add("IV")
            elif ds == -1:
                types.add("I")
            elif ds == 1:
                types.add("II")
            elif ds == 0:
                types.add("III")

    return types if types else {"None"}


# ============================================================
# Per-cell classification (from eval_vlm_fullscale.py lines 180-191)
# ============================================================

def failure_labels(delta_S_star, delta_S_obs, E_error):
    """
    Return the paper's `failures` string list, e.g.
    ['missed_exit_t2_e0', 'spurious_t1_e1'].

    Verbatim from eval_vlm_fullscale.py lines 180-191. Indices are
    1-based for the transition (t) and 0-based for the entity (e),
    matching the stored result records.
    """
    delta_S_star = np.asarray(delta_S_star)
    delta_S_obs = np.asarray(delta_S_obs)
    E_error = np.asarray(E_error)
    n_trans, n_ent = delta_S_star.shape

    failures = []
    for t in range(n_trans):
        for e in range(n_ent):
            if E_error[t, e] == 0:
                continue
            if delta_S_star[t, e] == 1 and delta_S_obs[t, e] != 1:
                failures.append(f"missed_entry_t{t+1}_e{e}")
            elif delta_S_star[t, e] == -1 and delta_S_obs[t, e] != -1:
                failures.append(f"missed_exit_t{t+1}_e{e}")
            elif delta_S_star[t, e] == 0 and delta_S_obs[t, e] != 0:
                failures.append(f"spurious_t{t+1}_e{e}")
    return failures


def classify_cells(delta_S_star, delta_S_obs, E_error, entity_ids=None):
    """
    Structured per-cell failure records:

        [{"type": "I", "type_name": "Context Bleeding",
          "transition_index": 1, "entity_index": 0, "entity_id": "A",
          "prescribed": -1, "observed": 0, "error": 1}, ...]

    Same cells as `failure_labels`, with Type IV (|E| = 2) resolved as in
    `classify_failures`.
    """
    delta_S_star = np.asarray(delta_S_star)
    delta_S_obs = np.asarray(delta_S_obs)
    E_error = np.asarray(E_error)
    n_trans, n_ent = delta_S_star.shape
    if entity_ids is None:
        entity_ids = [chr(ord("A") + j) for j in range(n_ent)]

    cells = []
    for t in range(n_trans):
        for e in range(n_ent):
            ee = int(E_error[t, e])
            if ee == 0:
                continue
            ds = int(delta_S_star[t, e])
            if abs(ee) == 2:
                ftype = "IV"
            elif ds == -1:
                ftype = "I"
            elif ds == 1:
                ftype = "II"
            else:
                ftype = "III"
            cells.append({
                "type": ftype,
                "type_name": TYPE_NAMES[ftype],
                "transition_index": t + 1,
                "entity_index": e,
                "entity_id": entity_ids[e],
                "prescribed": ds,
                "observed": int(delta_S_obs[t, e]),
                "error": ee,
            })
    return cells


if __name__ == "__main__":
    from itertools import product

    # Relay: A -> AB -> B, ΔS* = [[0, +1], [-1, 0]]
    delta_S_star = np.array([[0, 1], [-1, 0]])
    S_star = np.array([[1, 0], [1, 1], [0, 1]])

    cases = {
        "Perfect":                 [[1, 0], [1, 1], [0, 1]],
        "Type I  (A never exits)": [[1, 0], [1, 1], [1, 1]],
        "Type II (B never enters)": [[1, 0], [1, 0], [0, 0]],
        "Type III (spurious)":     [[1, 1], [1, 1], [0, 1]],
        "Type IV (reversed)":      [[1, 1], [1, 0], [0, 1]],
    }

    print("=" * 66)
    print("STEM Failure Taxonomy")
    print("=" * 66)
    for name, S_obs in cases.items():
        S_obs = np.array(S_obs)
        dS_obs = np.diff(S_obs, axis=0)
        E = dS_obs - delta_S_star
        types = sorted(classify_failures(delta_S_star, E))
        print(f"\n{name}")
        print(f"  S_obs = {S_obs.tolist()}   E = {E.tolist()}")
        print(f"  types  = {types} ({', '.join(TYPE_NAMES[t] for t in types)})")
        print(f"  labels = {failure_labels(delta_S_star, dS_obs, E)}")
