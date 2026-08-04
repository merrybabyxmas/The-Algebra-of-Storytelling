"""
SPAN: State Progression and Algebraic Narrative.

The algebraic layer of the framework: basis matrices, the completeness
theorem (Thm 3.1), the four composition operators, and the canonical
pattern library that defines MEMS-1K's prescribed matrices.
"""
from .basis_matrices import B1, B2, J, J_MATRICES
from .pattern_library import (
    PATTERNS,
    PATTERN_PRESENCE,
    PATTERN_DELTA,
    get_prescribed_mask,
    n_prescribed,
)
from .operators import (
    column_permute,
    time_reversal,
    spatial_expand,
    temporal_shift,
    compose_global,
)

__all__ = [
    "B1", "B2", "J", "J_MATRICES",
    "PATTERNS", "PATTERN_PRESENCE", "PATTERN_DELTA",
    "get_prescribed_mask", "n_prescribed",
    "column_permute", "time_reversal", "spatial_expand",
    "temporal_shift", "compose_global",
]
