"""
SPAN Framework: Canonical Pattern Library (MEMS-1K)

Defines the 7 narrative patterns with their prescribed presence
matrices S* and transition matrices ΔS*. Corresponds to Table S13.
"""
import numpy as np

PATTERNS = {
    "Relay": {
        "type": "Basis",
        "entities": 2,
        "notation": "A → AB → B",
        "S_star": np.array([[1, 0], [1, 1], [0, 1]]),
        "delta_S_star": np.array([[0, 1], [-1, 0]]),
        "scenarios": 200,
    },
    "Split": {
        "type": "Basis",
        "entities": 2,
        "notation": "AB → A → B",
        "S_star": np.array([[1, 1], [1, 0], [0, 1]]),
        "delta_S_star": np.array([[0, -1], [-1, 1]]),
        "scenarios": 200,
    },
    "Accumulation": {
        "type": "Derived",
        "entities": 3,
        "notation": "A → AB → ABC",
        "S_star": np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1]]),
        "delta_S_star": np.array([[0, 1, 0], [0, 0, 1]]),
        "scenarios": 200,
    },
    "Convergence": {
        "type": "Derived",
        "entities": 2,
        "notation": "A → B → AB",
        "S_star": np.array([[1, 0], [0, 1], [1, 1]]),
        "delta_S_star": np.array([[-1, 1], [1, 0]]),
        "scenarios": 100,
    },
    "Sliding_Window": {
        "type": "Derived",
        "entities": 3,
        "notation": "AB → BC → C",
        "S_star": np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]]),
        "delta_S_star": np.array([[-1, 0, 1], [0, -1, 0]]),
        "scenarios": 100,
    },
    "Reduction": {
        "type": "Derived",
        "entities": 3,
        "notation": "ABC → AB → A",
        "S_star": np.array([[1, 1, 1], [1, 1, 0], [1, 0, 0]]),
        "delta_S_star": np.array([[0, 0, -1], [0, -1, 0]]),
        "scenarios": 100,
    },
    "Reverse_Relay": {
        "type": "Derived",
        "entities": 2,
        "notation": "B → AB → A",
        "S_star": np.array([[0, 1], [1, 1], [1, 0]]),
        "delta_S_star": np.array([[1, 0], [0, -1]]),
        "scenarios": 100,
    },
}


# ============================================================
# Flat lookups — the single source of truth for S* and ΔS*.
#
# These replace the dicts that used to be copy-pasted into
# eval_vlm_fullscale.py, vlm_detector.py and the rebuttal scripts.
# ============================================================

PATTERN_NAMES = list(PATTERNS.keys())

PATTERN_PRESENCE = {name: p["S_star"].tolist() for name, p in PATTERNS.items()}
PATTERN_DELTA = {name: p["delta_S_star"] for name, p in PATTERNS.items()}


def get_pattern(pattern_name):
    """Return the pattern record, with a helpful error for unknown names."""
    if pattern_name not in PATTERNS:
        raise KeyError(
            f"Unknown pattern {pattern_name!r}. "
            f"Expected one of {PATTERN_NAMES}"
        )
    return PATTERNS[pattern_name]


def get_prescribed_mask(pattern_name):
    """Return the binary mask M where ΔS* ≠ 0 (prescribed transitions)."""
    dS = PATTERNS[pattern_name]["delta_S_star"]
    return (dS != 0).astype(int)


def n_prescribed(pattern_name):
    """Number of prescribed transitions (non-zero cells in ΔS*)."""
    return get_prescribed_mask(pattern_name).sum()


if __name__ == "__main__":
    print("=== MEMS-1K Pattern Library ===\n")
    total = 0
    for name, p in PATTERNS.items():
        n_p = n_prescribed(name)
        ptm_values = [round(1 - k / n_p, 2) for k in range(n_p + 1)]
        total += p["scenarios"]
        print(f"{name} ({p['type']}, {p['entities']}-ent, n={p['scenarios']})")
        print(f"  {p['notation']}")
        print(f"  ΔS* = {p['delta_S_star'].tolist()}")
        print(f"  Prescribed cells: {n_p}, PTM ∈ {ptm_values}")
        print()
    print(f"Total scenarios: {total}")
